

import os, random, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from datasets import Dataset, ClassLabel
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, get_cosine_schedule_with_warmup,
    default_data_collator
)

CSV_PATH   = r"C:\Users\C\Desktop\MBTI\mbti_cleaned.csv"
MODEL_NAME = "beomi/KcELECTRA-base"
EPOCHS     = 5
LR         = 1e-5
MAX_LEN    = 128
BATCH_TRAIN= 16
BATCH_EVAL = 16
VAL_RATIO  = .2
GAMMA      = 2.0
SEED       = 42           

warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

df = (pd.read_csv(CSV_PATH)
        .dropna(subset=["a_mbti", "text"])
        .reset_index(drop=True))

for i, c in enumerate(["ei", "ns", "ft", "pj"]):
    df[c] = df["a_mbti"].str[i].str.lower()

print("Total :", len(df))      #  88,418개 들어가야함

dims     = ["ei", "ns", "ft", "pj"]
encoders = {d: LabelEncoder().fit(df[d]) for d in dims}
for d in dims:
    df[d] = encoders[d].transform(df[d])

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(batch):
    return tok(batch["text"], truncation=True,
               padding="max_length", max_length=MAX_LEN)

class FocalTrainer(Trainer):
    def __init__(self, weights, gamma, *a, **kw):
        super().__init__(*a, **kw); self.w, self.g = weights.to(self.args.device), gamma
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        y = inputs.pop("labels"); logit = model(**inputs).logits
        ce = cross_entropy(logit, y, weight=self.w, reduction="none")
        loss = ((1-torch.exp(-ce))**self.g * ce).mean()
        return (loss, logit) if return_outputs else loss
    def create_optimizer_and_scheduler(self, num_training_steps: int, **kw):
        super().create_optimizer_and_scheduler(num_training_steps=num_training_steps)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, int(0.1*num_training_steps), num_training_steps)

cache_dir = Path("tmp_preds"); cache_dir.mkdir(exist_ok=True)
pred_map  = {}

for d in dims:
    cache_file = cache_dir / f"{d}.npy"
    if cache_file.exists():
        ids = np.load(cache_file, allow_pickle=False)
        pred_map[d] = encoders[d].inverse_transform(ids)
        print(f"{d} 축 캐시 ({len(ids)})"); continue

    print(f"\n── {d} 학습")
    raw   = Dataset.from_pandas(df[["text", d]].rename(columns={d:"label"}))
    raw   = raw.cast_column("label", ClassLabel(num_classes=2))
    split = raw.train_test_split(test_size=VAL_RATIO, seed=SEED,
                                 stratify_by_column="label")
    tr_ds = split["train"].map(tokenize, batched=True).remove_columns("text").with_format("torch")
    va_ds = split["test"] .map(tokenize, batched=True).remove_columns("text").with_format("torch")

    cls_w = torch.tensor(
        compute_class_weight('balanced',
                             classes=np.unique(tr_ds["label"]),
                             y=np.array(tr_ds["label"])),
        dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=2).to(device)

    args = TrainingArguments(
        output_dir=f"tmp_{d}",
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        eval_steps=max(1, len(tr_ds)//EPOCHS),
        do_eval=True,
        save_strategy="no",
        seed=SEED,
        logging_steps=100,
        dataloader_drop_last=False,
        fp16=False,          
    )

    trainer = FocalTrainer(cls_w, GAMMA, model=model, args=args,
                           train_dataset=tr_ds, eval_dataset=va_ds)
    trainer.train()

    full_ds = raw.map(tokenize, batched=True).remove_columns("text").with_format("torch")
    loader  = DataLoader(full_ds, batch_size=BATCH_EVAL,
                         shuffle=False, collate_fn=default_data_collator,
                         drop_last=False, pin_memory=True)

    model.eval(); ids = []
    with torch.no_grad():
        for batch in loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            ids.append(model(**batch).logits.argmax(1).cpu().numpy())
    ids = np.concatenate(ids)
    print(f"{d} 예측 완료 :", len(ids))

    np.save(cache_file, ids)
    pred_map[d] = encoders[d].inverse_transform(ids)

df["pred"] = ["".join(t) for t in zip(*(pred_map[k] for k in dims))]
df[["a_mbti","pred","text"]].to_csv("mbti_pred.csv", index=False, encoding="utf-8-sig")
print("결과 저장: mbti_pred.csv")

import matplotlib; matplotlib.use("Agg")
cnt = df["pred"].value_counts().sort_index()
plt.figure(figsize=(10,4))
plt.bar(cnt.index, cnt.values)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("mbti_dist.png")


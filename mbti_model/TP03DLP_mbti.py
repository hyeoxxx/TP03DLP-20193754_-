#!/usr/bin/env python
# mbti_train_cuda.py   (Windows + CUDA GPU)

import os, random, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from datasets import Dataset, ClassLabel
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    default_data_collator,
)
import re


CSV_PATH = r"C:\Users\C\Desktop\MBTI\mbti_cleaned.csv"
MODEL_NAME = "beomi/KcELECTRA-base"
TARGET_PER_TYPE = 800  # ★
EPOCHS = 5
LR = 1e-5
MAX_LEN = 256
BATCH_TRAIN = 16
BATCH_EVAL = 16
VAL_RATIO = 0.2
GAMMA = 2.0
SEED = 42

warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


df = pd.read_csv(CSV_PATH).dropna(subset=["a_mbti", "text"]).reset_index(drop=True)

for i, c in enumerate(["ei", "ns", "ft", "pj"]):
    df[c] = df["a_mbti"].str[i].str.lower()
df["type"] = df["a_mbti"].str.lower()

df = (
    df.groupby("type", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), TARGET_PER_TYPE), random_state=SEED))
    .reset_index(drop=True)
)
print("Sample size:", len(df))  # 16 × 800 = 12 800? 맞음

dims = ["ei", "ns", "ft", "pj"]
encoders = {d: LabelEncoder().fit(df[d]) for d in dims}
for d in dims:
    df[d] = encoders[d].transform(df[d])


tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)


class FocalTrainer(Trainer):
    def __init__(self, weights, gamma, *args, **kw):
        super().__init__(*args, **kw)
        self.w, self.g = weights.to(self.args.device), gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        y = inputs.pop("labels")
        logits = model(**inputs).logits
        ce = cross_entropy(logits, y, weight=self.w, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.g * ce).mean()
        return (loss, logits) if return_outputs else loss

    def create_optimizer_and_scheduler(self, num_training_steps: int, **kwargs):
        super().create_optimizer_and_scheduler(num_training_steps=num_training_steps)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps,
        )


cache_dir = Path("tmp_preds")
cache_dir.mkdir(exist_ok=True)
pred_map = {}

for d in dims:
    cache_file = cache_dir / f"{d}.npy"

    if cache_file.exists(): # 예측된 축이면 로드하는 코드
        ids = np.load(cache_file)
        pred_map[d] = encoders[d].inverse_transform(ids)
        continue

    print(f"\n── {d} 진행")
    raw = Dataset.from_pandas(df[["text", d]].rename(columns={d: "label"}))
    raw = raw.cast_column("label", ClassLabel(num_classes=2))
    split = raw.train_test_split(
        test_size=VAL_RATIO, seed=SEED, stratify_by_column="label"
    )

    tr_ds = (
        split["train"]
        .map(tokenize, batched=True)
        .remove_columns("text")
        .with_format("torch")
    )
    va_ds = (
        split["test"]
        .map(tokenize, batched=True)
        .remove_columns("text")
        .with_format("torch")
    )

    cls_w = torch.tensor(
        compute_class_weight(
            "balanced", classes=np.unique(tr_ds["label"]), y=np.array(tr_ds["label"])
        ),
        dtype=torch.float,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    args = TrainingArguments(
        output_dir=f"tmp_{d}",
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        eval_steps=max(1, len(tr_ds) // EPOCHS),
        do_eval=True,
        save_strategy="no",
        seed=SEED,
        logging_steps=100,
        dataloader_drop_last=False,
        fp16=True,
    )

    trainer = FocalTrainer(
        cls_w, GAMMA, model=model, args=args, train_dataset=tr_ds, eval_dataset=va_ds
    )
    trainer.train()

    model.save_pretrained(f"saved_{d}")
    tok.save_pretrained(f"saved_{d}")

    full_ds = (
        raw.map(tokenize, batched=True).remove_columns("text").with_format("torch")
    )
    loader = DataLoader(
        full_ds,
        batch_size=BATCH_EVAL,
        shuffle=False,
        collate_fn=default_data_collator,
        drop_last=False,
    )

    model.eval()
    batches = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batches.append(model(**batch).logits.argmax(1).cpu().numpy())

    ids = np.concatenate(batches)
    print(f"{d} 예측 완료 :", len(ids))

    if len(ids) != len(df):
        raise RuntimeError(f"{d} 길이 불일치 {len(ids)} vs {len(df)}")

    np.save(cache_file, ids)  # 캐시 저장
    pred_map[d] = encoders[d].inverse_transform(ids)  # 최종 예측값 문자화해서 저장

for k in dims:
    print(f"{k} 랭스 :", len(pred_map[k]))
df["pred"] = ["".join(tup) for tup in zip(*(pred_map[k] for k in dims))]
df[["a_mbti", "pred", "text"]].to_csv(
    "mbti_pred.csv", index=False, encoding="utf-8-sig"
)
print("결과 저장: mbti_pred.csv")

import matplotlib

matplotlib.use("Agg")
cnt = df["pred"].value_counts().sort_index()
plt.figure(figsize=(10, 4))
plt.bar(cnt.index, cnt.values)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("mbti_dist.png")

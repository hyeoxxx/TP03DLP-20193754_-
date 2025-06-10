import pandas as pd # type: ignore

df = pd.read_csv(r"C:\Users\C\Desktop\MBTI\multiple_qna_cleaned.tsv", sep="\t")

df["question"] = df["question"].fillna("").astype(str)
df["answer"]   = df["answer"].fillna("").astype(str)

def clean_text(text):
    return text.replace("[SEP]", " ").strip()

#df["text"] = (df["question"] + " " + df["answer"]).apply(clean_text)
df["text"] = df["answer"].astype(str).apply(clean_text)

print(df[["a_mbti", "text"]].head())

df[["a_mbti", "text"]].to_csv("preprocessed_mbti.csv", index=False, encoding="utf-8-sig")

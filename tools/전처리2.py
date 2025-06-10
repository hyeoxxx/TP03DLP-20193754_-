import pandas as pd
import re

MIN_LEN = 10

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text)  # 연속 공백 제거
    text = text.strip()
    return text

df = pd.read_csv(r"C:\Users\C\Desktop\MBTI\mbti.csv")

df["a_mbti"] = df["a_mbti"].str.lower().str.strip()
df["text"] = df["text"].astype(str).apply(clean_text)

df = df[df["text"].str.len() >= MIN_LEN]

df = df[df["text"].str.strip() != ""]

df = df.drop_duplicates(subset="text")


print(f"완료 데이터 수: {len(df)}")

df.to_csv("mbti_cleaned.csv", index=False, encoding="utf-8-sig")

import time, json, urllib.parse, jmespath, pandas as pd
from pathlib import Path
from playwright.sync_api import sync_playwright

MBTIS = ["ENTP","ENTJ","ENFP","ENFJ","ESTP","ESTJ","ESFP","ESFJ",
         "INTP","INTJ","INFP","INFJ","ISTP","ISTJ","ISFP","ISFJ"]
KWDS  = [
    ("영화", "추천"),
    ("음악", "추천"),
    ("좋아하는", "영화"),
    ("좋아하는", "노래")
]
MAX_T  = 200
SCROLL_TIMEOUT = 30
OUT    = Path("mbti_tweets_playwright.csv")

TWEET_GQL = "TweetResult"

def build_query(mbti, kw_pair):
    kw1, kw2 = kw_pair
    
    return f'({mbti} OR #{mbti}) "{kw1}" "{kw2}" lang:ko -is:retweet'

def extract_tweet(js):
    tw = jmespath.search("data.tweetResult.result.legacy", js)
    if not tw: return None
    return {
        "tweet_id": tw["id_str"],
        "date":     tw["created_at"],
        "text":     tw["full_text"],
        "fav":      tw["favorite_count"],
        "rt":       tw["retweet_count"]
    }

rows = []

with sync_playwright() as p:
    br   = p.firefox.launch(headless=True)
    page = br.new_page()

    for mbti in MBTIS:
        for kw in KWDS:
            q_raw  = build_query(mbti, kw)
            enc_q  = urllib.parse.quote(q_raw, safe="")
            print(f"{q_raw}")

            tweets = {}
            def h(resp):
                if TWEET_GQL in resp.url and resp.ok:
                    try:
                        js = resp.json()
                        tw = extract_tweet(js)
                        if tw: tweets.setdefault(tw["tweet_id"], tw)
                    except: pass

            page.on("response", h)
            page.goto(f"https://x.com/search?q={enc_q}&f=live", timeout=0)

            start = time.time()
            while len(tweets) < MAX_T and time.time() - start < SCROLL_TIMEOUT:
                page.mouse.wheel(0, 2500)
                time.sleep(1)

            for tw in tweets.values():
                rows.append({
                    "mbti": mbti,
                    "kw1": kw[0], "kw2": kw[1],
                    "date": tw["date"], "text": tw["text"],
                    "fav":  tw["fav"],  "rt":  tw["rt"],
                    "url": f"https://x.com/i/web/status/{tw['tweet_id']}"
                })
            page.remove_listener("response", h)
            time.sleep(2)

pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
print(f" {len(rows)} rows → {OUT.resolve()} saved")


#
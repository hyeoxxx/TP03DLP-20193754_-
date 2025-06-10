
import ssl, time, snscrape.modules.twitter as sntwitter, pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context

mbti_types = ["INFP", "ENFP", "INTJ", "ISTP", "ESFP", "ENTP", "ISFJ", "ENTJ"]
keywords   = ["영화 추천", "음악 추천", "좋아하는 영화", "좋아하는 노래"]

max_tweets = 200
results    = []

for mbti in mbti_types:
    for kw in keywords:
        query = f'"{mbti}" "{kw}" lang:ko'
        print(f"🔍 {query}")
        try:
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
                if i >= max_tweets:
                    break
                results.append(
                    dict(mbti=mbti, keyword=kw,
                         username=tweet.user.username,
                         date=tweet.date, text=tweet.content,
                         url=tweet.url)
                )
                time.sleep(0.2)     # 타임아웃
        except Exception as e:
            print("exception : ", e)

pd.DataFrame(results).to_csv("mbti_tweets_ssl_bypass.csv",
                             index=False, encoding="utf-8-sig")
print(f"{len(results)} rows saved")

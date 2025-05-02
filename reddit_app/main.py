from fastapi import FastAPI, HTTPException
import requests
import os

app = FastAPI(title="Reddit API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Reddit API"}

@app.get("/reddit/{subreddit}")
def get_reddit(subreddit: str, limit: int = 10):
    url = f"https://www.reddit.com/r/{subreddit}/top.json?limit={limit}"
    headers = {"User-Agent": "FastAPI Reddit API"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        posts = data["data"]["children"]
        
        results = []
        for post in posts:
            post_data = post["data"]
            results.append({
                "title": post_data["title"],
                "score": post_data["score"],
                "url": post_data["url"],
                "author": post_data["author"]
            })
        
        return {"subreddit": subreddit, "posts": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
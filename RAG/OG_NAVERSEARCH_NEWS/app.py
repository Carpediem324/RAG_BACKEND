import os
import urllib.request
import urllib.parse
import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()  # Load environment variables

app = FastAPI()

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API keys configuration
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
    raise EnvironmentError("NAVER_CLIENT_ID or NAVER_CLIENT_SECRET is not set in the environment.")

class MessageRequest(BaseModel):
    message: str

async def search_naver(query: str):
    enc_text = urllib.parse.quote(query)
    url = f"https://openapi.naver.com/v1/search/news.json?query={enc_text}"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", NAVER_CLIENT_ID)
    request.add_header("X-Naver-Client-Secret", NAVER_CLIENT_SECRET)
    try:
        response = urllib.request.urlopen(request)
        rescode = response.getcode()
        if rescode == 200:
            response_body = response.read().decode('utf-8')
            data = json.loads(response_body)  # JSON 파싱
            items = data.get("items", [])
            results = [
                {"title": item["title"], "link": item["link"]} 
                for item in items if "title" in item and "link" in item
            ]
            return results
        else:
            return []
    except Exception as e:
        print(f"Error occurred: {e}")
        return []

@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    # Try Naver Search
    naver_result = await search_naver(req.message)
    if naver_result:
        return {
            "reply": "Results found from Naver Search API",
            "results": naver_result
        }
    else:
        return {
            "reply": "No results found from Naver Search API",
            "results": []
        }

@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

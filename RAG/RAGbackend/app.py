import os
import urllib.request
import urllib.parse
import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from serpapi import GoogleSearch
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_upstage import ChatUpstage

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
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

if not UPSTAGE_API_KEY:
    raise EnvironmentError("UPSTAGE_API_KEY is not set in the environment.")

if not SERP_API_KEY:
    raise EnvironmentError("SERP_API_KEY is not set in the environment.")

if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
    raise EnvironmentError("NAVER_CLIENT_ID or NAVER_CLIENT_SECRET is not set in the environment.")

chat_upstage = ChatUpstage()

class MessageRequest(BaseModel):
    message: str

async def extract_keywords(message: str):
    # Use LLM to extract keywords for Naver
    naver_template = "Extract the most relevant single keyword from the following question to use in a search query:\n\nQuestion: {question}\nKeyword:"
    prompt = ChatPromptTemplate.from_template(naver_template)
    chain = prompt | chat_upstage | StrOutputParser()
    keyword = chain.invoke({"question": message})
    return keyword.strip()

async def refine_query_for_google(message: str):
    # Use LLM to create a more descriptive query for Google
    google_template = "Refine the following question to make it suitable for a web search query:\n\nQuestion: {question}\nRefined Query:"
    prompt = ChatPromptTemplate.from_template(google_template)
    chain = prompt | chat_upstage | StrOutputParser()
    refined_query = chain.invoke({"question": message})
    return refined_query.strip()

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
            data = json.loads(response_body)
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

async def search_google(query: str):
    params = {
        "engine": "google",
        "q": query,
        "num": "4",
        "api_key": SERP_API_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict().get("organic_results", [])
    return [
        {"title": result.get("title"), "link": result.get("link")}
        for result in results
    ]

@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    # Step 1: Extract Keywords for Naver
    naver_keyword = await extract_keywords(req.message)

    # Step 2: Try Naver Search
    naver_results = await search_naver(naver_keyword)
    source = "Naver Search API"

    # Step 3: Fallback to Google Search if no Naver result
    if not naver_results:
        google_query = await refine_query_for_google(req.message)
        naver_results = await search_google(google_query)
        source = "Google Search API"

    if not naver_results:
        return {"reply": "No results found from both Naver and Google.", "source": "None"}

    # Step 4: Prepare context for RAG
    context = "\n\n".join([f"{result['title']}\n{result['link']}" for result in naver_results])

    # Step 5: Generate Answer using RAG
    template = """Answer the question based on context.\n\nQuestion: {question}\nContext: {context}\nAnswer:"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"question": RunnablePassthrough(), "context": RunnablePassthrough()}
        | prompt
        | chat_upstage
        | StrOutputParser()
    )
    result = chain.invoke({"question": req.message, "context": context})
    result = result.split(":")[1].strip() if ":" in result else result.strip()

    return {"reply": result, "source": source, "results": naver_results}

@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

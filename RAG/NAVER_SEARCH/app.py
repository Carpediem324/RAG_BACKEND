import os
import urllib.request
import urllib.parse
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
            return response_body  # Return raw JSON
        else:
            return None
    except Exception:
        return None

async def search_google(query: str):
    params = {
        "engine": "google",
        "q": query,
        "num": "4",
        "api_key": SERP_API_KEY
    }
    search = GoogleSearch(params)
    return search.get_dict()

@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    # Step 1: Try Naver Search
    naver_result = await search_naver(req.message)
    urls = []

    if naver_result:
        import json
        naver_data = json.loads(naver_result)
        if 'items' in naver_data:
            urls = [item['link'] for item in naver_data['items']]

    # Step 2: Fallback to Google Search if no Naver result
    if not urls:
        google_result = await search_google(req.message)
        urls = [result['link'] for result in google_result.get('organic_results', [])]

    if not urls:
        return {"reply": "No results found from both Naver and Google."}

    # Step 3: Load URLs and extract content
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    # Step 4: LangChain setup for RAG
    template = """Answer the question based on context.

    Question: {question}
    Context: {context}
    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()

    chain = (
        {"question": RunnablePassthrough(), "context": RunnablePassthrough()}
        | prompt
        | chat_upstage
        | parser
    )

    # Step 5: Split documents for efficient processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(data)
    docs = [doc.page_content for doc in splits]

    # Step 6: Generate answer
    context = "\n\n".join(docs)
    result = chain.invoke({"question": req.message, "context": context})
    result = result.split(":")[1].strip() if ":" in result else result.strip()

    return {"reply": result}

@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

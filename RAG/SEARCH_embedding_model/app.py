import os
import urllib.request
import urllib.parse
import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from serpapi import GoogleSearch

# LangChain, Upstage, Chroma 관련
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ▶ 추가된 부분
from langchain_chroma import Chroma
from langchain_upstage import ChatUpstage, UpstageEmbeddings

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

# LLM 초기화
chat_upstage = ChatUpstage(api_key=UPSTAGE_API_KEY)

class MessageRequest(BaseModel):
    message: str

# 1) 네이버 검색어 추출
async def extract_keywords(message: str):
    naver_template = (
        "Extract the most relevant single keyword from the following question "
        "to use in a search query:\n\nQuestion: {question}\nKeyword:"
    )
    prompt = ChatPromptTemplate.from_template(naver_template)
    chain = prompt | chat_upstage | StrOutputParser()
    keyword = chain.invoke({"question": message})
    return keyword.strip()

# 2) 구글 검색어 리파인
async def refine_query_for_google(message: str):
    google_template = (
        "Refine the following question to make it suitable for a web search query:\n\n"
        "Question: {question}\nRefined Query:"
    )
    prompt = ChatPromptTemplate.from_template(google_template)
    chain = prompt | chat_upstage | StrOutputParser()
    refined_query = chain.invoke({"question": message})
    return refined_query.strip()

# 3) 네이버 검색
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
            response_body = response.read().decode("utf-8")
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

# 4) 구글 검색
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
    # Step 1. 네이버 검색을 위한 단일 키워드 추출
    naver_keyword = await extract_keywords(req.message)

    # Step 2. 네이버 뉴스 검색
    naver_results = await search_naver(naver_keyword)
    source = "Naver Search API"

    # Step 3. 네이버 검색 결과가 없다면 구글 검색
    if not naver_results:
        google_query = await refine_query_for_google(req.message)
        naver_results = await search_google(google_query)
        source = "Google Search API"

    # 아무 검색결과도 없을 때
    if not naver_results:
        return {"reply": "No results found from both Naver and Google.", "source": "None"}

    # ----------------------------------------------------------
    # ★★★ 여기가 핵심: 검색 결과 → 벡터스토어(Chroma) → top-k 추출
    # ----------------------------------------------------------

    # (1) 검색 결과를 Document 형식으로 변환
    #     LangChain의 Document(page_content=..., metadata=...)를 사용하는 것이 일반적
    documents = []
    for item in naver_results:
        content = f"{item['title']}\n{item['link']}"
        documents.append(Document(page_content=content))

    # (2) 문서를 Chunking - 제목/링크 뿐이라 짧을 수 있지만 실제 본문이 있다고 가정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(documents)

    # (3) 임베딩 생성 (UpstageEmbeddings 사용)
    embeddings = UpstageEmbeddings(model="embedding-query", api_key=UPSTAGE_API_KEY)

    # (4) Chroma 벡터스토어 생성
    chroma_db = Chroma.from_documents(doc_splits, embedding=embeddings)

    # (5) Retriever 객체를 통해 top-k 추출
    retriever = chroma_db.as_retriever(
        search_type="mmr",  # or similarity
        search_kwargs={"k": 3}
    )
    relevant_docs = retriever.get_relevant_documents(req.message)

    # (6) 최종 context로 사용할 문자열 생성
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    # ----------------------------------------------------------

    # Step 4. RAG 파이프라인으로 답변 생성
    template = """Answer the question based on context.

    Question: {question}
    Context: {context}
    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"question": RunnablePassthrough(), "context": RunnablePassthrough()}
        | prompt
        | chat_upstage
        | StrOutputParser()
    )
    result = chain.invoke({"question": req.message, "context": context})

    # ":" 이후를 자르는 간단 파싱
    result = result.split(":")[1].strip() if ":" in result else result.strip()

    # 반환
    return {
        "reply": result,
        "source": source,
        "results": [
            {
                "title": d.page_content.split("\n")[0],
                "link": d.page_content.split("\n")[1] if "\n" in d.page_content else ""
            }
            for d in relevant_docs
        ]
    }

@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

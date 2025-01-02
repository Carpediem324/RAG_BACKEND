import os
import urllib.request
import urllib.parse
import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from serpapi import GoogleSearch

# LangChain 관련
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Chroma, UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_upstage import ChatUpstage, UpstageEmbeddings

load_dotenv()  # Load environment variables

app = FastAPI()

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

if not UPSTAGE_API_KEY:
    raise EnvironmentError("UPSTAGE_API_KEY is not set in the environment.")

if not SERP_API_KEY:
    raise EnvironmentError("SERP_API_KEY is not set in the environment.")

if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
    raise EnvironmentError("NAVER_CLIENT_ID or NAVER_CLIENT_SECRET is not set.")

chat_upstage = ChatUpstage(api_key=UPSTAGE_API_KEY)

class MessageRequest(BaseModel):
    message: str

# (1) 네이버 검색어 추출
async def extract_keywords(message: str):
    naver_template = (
        "Extract the most relevant single keyword from the following question "
        "to use in a search query:\n\n"
        "Question: {question}\nKeyword:"
    )
    prompt = ChatPromptTemplate.from_template(naver_template)
    chain = prompt | chat_upstage | StrOutputParser()
    keyword = chain.invoke({"question": message})
    return keyword.strip()

# (2) 구글 검색 질의 확장
async def refine_query_for_google(message: str):
    google_template = (
        "Refine the following question to make it suitable for a web search query:\n\n"
        "Question: {question}\nRefined Query:"
    )
    prompt = ChatPromptTemplate.from_template(google_template)
    chain = prompt | chat_upstage | StrOutputParser()
    refined_query = chain.invoke({"question": message})
    return refined_query.strip()

# (3) 네이버 뉴스 검색 (description 필드 활용)
async def search_naver(query: str):
    enc_text = urllib.parse.quote(query)
    url = f"https://openapi.naver.com/v1/search/news.json?query={enc_text}"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", NAVER_CLIENT_ID)
    request.add_header("X-Naver-Client-Secret", NAVER_CLIENT_SECRET)
    try:
        response = urllib.request.urlopen(request)
        if response.getcode() == 200:
            data = json.loads(response.read().decode("utf-8"))
            items = data.get("items", [])
            results = []
            for item in items:
                # title, originallink, description, pubDate, link 등
                title = item.get("title", "")
                link = item.get("link", "")
                desc = item.get("description", "")
                # 필요하다면 HTML 태그 제거 로직을 넣어도 됨
                # ex) re.sub(r'<[^>]+>', '', desc)
                results.append({
                    "title": title,
                    "link": link,
                    "description": desc
                })
            return results
        else:
            return []
    except Exception as e:
        print(f"Error in Naver search: {e}")
        return []

# (4) 구글 검색
async def search_google(query: str):
    params = {
        "engine": "google",
        "q": query,
        "num": "4",
        "api_key": SERP_API_KEY
    }
    search = GoogleSearch(params)
    data = search.get_dict()
    results = data.get("organic_results", [])
    return [
        {
            "title": r.get("title", ""),
            "link": r.get("link", ""),
            "description": r.get("snippet", "")  # 구글은 snippet으로 간단한 요약을 제공
        }
        for r in results
    ]

@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    # 1) 네이버용 키워드 추출
    naver_keyword = await extract_keywords(req.message)

    # 2) 네이버 뉴스 검색
    naver_results = await search_naver(naver_keyword)
    source = "Naver Search API"

    # 3) 네이버 결과가 없으면 구글 검색
    if not naver_results:
        google_query = await refine_query_for_google(req.message)
        naver_results = await search_google(google_query)
        source = "Google Search API"

    if not naver_results:
        return {"reply": "No results found from both Naver and Google.", "source": "None"}

    # ---------------------------------------------------------
    # description 필드 → 문서화 → chunking → 벡터스토어
    # ---------------------------------------------------------
    from langchain.docstore.document import Document

    documents = []
    for item in naver_results:
        # 실제 임베딩될 컨텐츠는 description
        # title, link는 메타데이터에 넣어두면 결과에서 확인 가능
        desc = item["description"]
        title = item["title"]
        link = item["link"]

        doc = Document(
            page_content=desc,
            metadata={
                "title": title,
                "link": link
            }
        )
        documents.append(doc)

    # 문서 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    # 임베딩 생성 (UpstageEmbeddings)
    embeddings = UpstageEmbeddings(model="embedding-query", api_key=UPSTAGE_API_KEY)

    # Chroma DB 구성
    vectorstore = Chroma.from_documents(split_docs, embedding=embeddings)

    # top-k 문서 검색
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":3})
    relevant_docs = retriever.get_relevant_documents(req.message)

    # 이제 RAG를 위해 context 생성
    # description을 벡터화했으나, 답변을 구성할 때 title/link를 포함할 수도 있음
    # 여기서는 단순히 description만 context로 합친 예시
    context = ""
    for d in relevant_docs:
        # d.page_content == description
        # d.metadata["title"], d.metadata["link"]
        context += f"Title: {d.metadata.get('title')}\n"
        context += f"Link: {d.metadata.get('link')}\n"
        context += f"Description: {d.page_content}\n\n"

    # RAG 파이프라인
    template = """You are a friendly and knowledgeable AI assistant that helps answer user questions using real-time news information. 

    - Use the provided 'Context' (top-k search results) to find relevant details. 
    - If the context doesn't contain enough information or you're unsure, express that politely.
    - Always explain in a clear, conversational style.

    Question: {question}

    Context:
    {context}

    Now provide a helpful, concise, and chatty answer to the user's question:
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"question": RunnablePassthrough(), "context": RunnablePassthrough()}
        | prompt
        | chat_upstage
        | StrOutputParser()
    )
    result = chain.invoke({"question": req.message, "context": context})

    # 간단 파싱
    if ":" in result:
        result = result.split(":")[-1].strip()

    return {
        "reply": result,
        "source": source,
        "results": [
            {
                "title": d.metadata.get("title", ""),
                "link": d.metadata.get("link", ""),
                "description": d.page_content
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

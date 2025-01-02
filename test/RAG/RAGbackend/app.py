import os
from typing import List
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

if not UPSTAGE_API_KEY:
    raise EnvironmentError("UPSTAGE_API_KEY is not set in the environment.")

if not SERP_API_KEY:
    raise EnvironmentError("SERP_API_KEY is not set in the environment.")

chat_upstage = ChatUpstage()

class MessageRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    # SERPAPI Search Setup
    params = {
        "engine": "google",
        "q": req.message,
        "num": "4",
        "api_key": SERP_API_KEY
    }
    search = GoogleSearch(params)
    search_result = search.get_dict()

    # Load URLs and extract content
    urls = [result['link'] for result in search_result['organic_results']]
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    # LangChain setup for RAG
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

    # Split documents for efficient processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(data)
    docs = [doc.page_content for doc in splits]

    # Generate answer
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

# -*- coding: utf-8 -*-
# 필요한 라이브러리 설치
#pip install -qU openai langchain langchain-upstage langchain-chroma getpass4 langchain-community unstructured google-search-results
#
import os
import getpass
import warnings
from serpapi import GoogleSearch
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_upstage import ChatUpstage

# 경고 무시\warnings.filterwarnings("ignore")

# Upstage API 키 설정
if "UPSTAGE_API_KEY" not in os.environ or not os.environ["UPSTAGE_API_KEY"]:
    os.environ["UPSTAGE_API_KEY"] = getpass.getpass("Enter your Upstage API key: ")

# SERPAPI API 키 설정
if not os.getenv("SERPAPI_API_KEY"):
    GoogleSearch.SERP_API_KEY = getpass.getpass("Enter your SERPAPI API key: ")
    os.environ["SERPAPI_API_KEY"] = GoogleSearch.SERP_API_KEY

# 질문 리스트
questions = [
    "What distinguishes the o1 model’s reasoning capabilities from previous OpenAI models, and how does 'chain-of-thought' improve its performance?",
    "How does the o1 model perform in jailbreak evaluations compared to GPT-4o, and what measures have been implemented to resist adversarial prompts?",
    "What are the key improvements in the o1 model regarding hallucinations and fairness compared to GPT-4o, and how were these measured?",
    "What safety challenges arise from o1’s enhanced reasoning abilities, and how does OpenAI address the risks associated with chain-of-thought reasoning?",
    "What were the findings from external red teaming efforts, particularly concerning the o1 model’s susceptibility to manipulation, persuasion, and scheming behaviors?"
]

# SERPAPI 검색 설정
params = {
    "engine": "google",
    "q": questions[0],
    "num": "4"
}
search = GoogleSearch(params)
search_result = search.get_dict()

# 수집된 URL로부터 데이터 로드
urls = [result['link'] for result in search_result['organic_results']]
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

# LangChain을 활용한 질의응답 체인 설정
template = """Answer the question based on context.

Question: {question}
Context: {context}
Answer:"""

llm = ChatUpstage()
prompt = ChatPromptTemplate.from_template(template)
parser = StrOutputParser()

chain = (
    {"question": RunnablePassthrough(), "context": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

# 첫 번째 질문 실행
print(f"질문: {questions[0]}")
print("-" * 20)
print(f"답변:\n{chain.invoke({'question': questions[0], 'context': data[0].page_content})}")

# 검색된 문서를 분할하여 검색 가능한 데이터로 변환
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
splits = text_splitter.split_documents(data)

# 분할된 문서 확인
print("Splits:", len(splits))

# 검색된 문서 기반으로 답변 생성
docs = [doc.page_content for doc in splits]
user_question = questions[0]

# Prompt 설정
system = """
Answer the question based on context.
"""
user = """
Question: {question}
Context: {context}

<<<Output Format>>>
`Answer: <Answer based on the document.>`
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", user),
])

rag_chain = prompt | llm | StrOutputParser()
generation = rag_chain.invoke({"context": "\n\n".join(docs), "question": user_question})
generation = generation.split(":")[1].strip() if ":" in generation else generation.strip()
print(generation)

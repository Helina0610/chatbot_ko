import os
from glob import glob
import re
from pprint import pprint

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import HuggingFaceEmbeddings

from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from langchain_huggingface import HuggingFacePipeline

# 파일 목록
pdf_files = glob(os.path.join('data', '*일상감사*.pdf'))

# pdf 파일을 읽어서 텍스트로 변환
loader = PyPDFLoader(pdf_files[0])
data = loader.load()


# HugoingFace Embeddings를 다운로드 (https://huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS)
embeddings_model = HuggingFaceEmbeddings(
    model_name="./local_model/KR-SBERT-V40K-klueNLI-augSTS",
)

query = "1건당 5,000만원을 초과하는 각종 용역업무는 일상감사를 받아야 하나요?"

vectorstore = Chroma.from_documents(documents=data, 
                                    embedding=embeddings_model, 
                                    collection_name="hongju_test",
                                    persist_directory="./chroma_db")

# query를 검색해서 k(5)개의 유사한 문서들을 가져옴
chroma_docs = vectorstore.similarity_search(query, k=5)
    


# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""


## RAG Chain
model_id="./local_model/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="auto",
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, top_k=50, temperature=0.1)
llm = HuggingFacePipeline(pipeline=pipe)

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 1}
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


response = rag_chain.invoke(query)
print(response)

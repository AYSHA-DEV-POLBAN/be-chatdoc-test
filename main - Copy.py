from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain_community.callbacks import get_openai_callback
import io

app = FastAPI()

knowledge_base = None
    
class Question(BaseModel):
    question: str


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # try:
        global knowledge_base  # Gunakan klausa global
        load_dotenv()
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_text(text)   
        
        # create embedding
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        return {"filename": file.filename, "content": text}

@app.post("/ask_question/")
async def ask_question(question: Question):
    if knowledge_base is None:
        return {"response": "Error! Knowledge base is not available yet. Please upload a file first."}
    
    load_dotenv()
    user_question = question.question
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    
    # Fetch documents from knowledge base
    docs = knowledge_base.similarity_search(user_question)
    
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        print(cb)
        
    return {"response": response}

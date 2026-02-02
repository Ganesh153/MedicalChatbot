"""
User Query to fetch that to our backend to get those LLM response
"""
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from pydantic import Field
from typing import List, Optional
from logger import logger
import os

router = APIRouter()

@router.post('/ask/')
async def ask_question(question:str=Form(...)):
    try:
        logger.info(f"User Query:{question}")

        #Embed model + Pinecone setup
        pc = Pinecone(api_key=os.environ("PINECONE_API_KEY"))
        index=pc.Index(os.environ("PINECONE_INDEX_NAME"))
        embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embedded_query=embed_model.embed_query(question)
        res=index.query(vector=embedded_query, top_k=3, include_metadata=True)

        docs=[
            Document(
                page_content=match["metadata"].get("text",""),
                metadata=match["metadata"]
            ) for match in res["matches"]
        ]

        # To get those relevant chunk extremely fast with extreme precise, we are defining this Simple retriever
        class SimpleRetriever(BaseRetriever):
            tags: Optional[List[str]] = Field(default_factory=list)
            metadata: Optional[dict] = Field(default_factory=dict)

            def __init__(self, documents: List[Document]):
                super().__init__()
                self._docs = documents

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self._docs
        
        retriever=SimpleRetriever(docs)
        chain=get_llm_chain(retriever)
        result = query_chain(chain, question)

        logger.info("Query result is successful")
        return result
    except Exception as e:
        logger.exception("Error Processing question")
        return JSONResponse(status_code=500, content={"error":str(e)})
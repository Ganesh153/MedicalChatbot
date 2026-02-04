from typing import List, Optional

from langchain_core.documents import Document

from logger import logger


def _format_sources(source_docs: Optional[List[Document]]) -> List[dict]:
    if not source_docs:
        return []

    formatted = []
    for doc in source_docs:
        metadata = doc.metadata or {}
        formatted.append(
            {
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "chunk_id": metadata.get("chunk_id"),
                "score": metadata.get("score"),
                "snippet": metadata.get("text", doc.page_content),
            }
        )
    return formatted


def query_chain(chain, user_input: str, source_docs: Optional[List[Document]] = None):
    try:
        logger.debug(f"Running chain for input: {user_input}")
        result = chain.invoke({"question": user_input})
        response = {
            "response": result.content if hasattr(result, "content") else str(result),
            "sources": _format_sources(source_docs),
        }
        logger.debug(f"Chain response: {response}")
        return response
    except Exception as e:
        logger.exception("Error on query chain")
        raise
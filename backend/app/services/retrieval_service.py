import logging
from typing import List, Dict, Any, Tuple, Optional 
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer, util
from backend.app.core.config import get_qdrant_client, GROQ_MODEL, GROQ_API_KEY, EMBED_MODEL_NAME
from backend.app.services.embedding_service import EmbeddingService

logger = logging.getLogger("retrieval_service")

try:
    rerank_model = SentenceTransformer(EMBED_MODEL_NAME)
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model for reranking: {e}")
    rerank_model = None

class RetrievalService:
    def __init__(self):
        self.qdrant_client = get_qdrant_client()
        self.embedding_service = EmbeddingService()

    def _rerank_chunks(self, query: str, hits: List[qmodels.ScoredPoint]) -> Tuple[List[qmodels.ScoredPoint], List[float]]:
        if not hits or rerank_model is None:
            return hits, [h.score for h in hits]
        try:
            query_embedding = rerank_model.encode(query, convert_to_tensor=True)
            text_embeddings = rerank_model.encode([h.payload.get("text", "") for h in hits], convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(query_embedding, text_embeddings)[0].tolist()
            paired = list(zip(hits, cosine_scores))
            paired.sort(key=lambda t: t[1], reverse=True)
            sorted_hits, sorted_scores = zip(*paired) if paired else ([], [])
            return list(sorted_hits), list(sorted_scores)
        except Exception as e:
            logger.exception(f"Reranker failed: {e}")
            return hits, [h.score for h in hits]

    def retrieve_and_answer(self, query: str, session_id: str, top_k: int = 20, final_n: int = 5, chat_history: Optional[List[Dict]]=None) -> Dict[str, Any]:
        collection_name = f"session_{session_id}"
        qvec = self.embedding_service._embed_texts([query])[0]
        
        try:
            hits = self.qdrant_client.search(collection_name=collection_name, query_vector=qvec, limit=top_k, with_payload=True)
        except Exception:
            return {"answer": "Please ingest documents before asking questions.", "supporting_chunks": []}
        
        if not hits:
            return {"answer": "I couldn't find anything relevant in your documents.", "supporting_chunks": []}

        reranked_hits, rerank_scores = self._rerank_chunks(query, hits)
        chosen_hits = reranked_hits[:final_n]
        
        
        #Build history and contex
        formatted_history = ""
        if chat_history:
            for message in chat_history:
                role = "User" if message["role"] == "user" else "Assistant"
                formatted_history += f"{role}: {message['content']}\n"
        
        
        ctx_lines, supporting_chunks = [], []
        for i, h in enumerate(chosen_hits, start=1):
            p = h.payload
            entry = {"rank": i, "rerank_score": rerank_scores[i-1], "doc_id": p.get("doc_id"), "page": p.get("page"),"para": p.get("para"),"theme":p.get("theme"), "text": p.get("text")}
            supporting_chunks.append(entry)
            ctx_lines.append(f"[C{i}] From '{entry['doc_id']}':\n{entry['text']}\n")
        context = "\n---\n".join(ctx_lines)
        
        try:
            truncated_context = context[:7000]
            llm = ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=0.1)
            
            #History aware prompt


            
            prompt = f"""
            You are an expert research assistant. Answer the user's new query based on the provided conversation history and the new context retrieved from documents.
            If the user is asking a follow-up question, use the history to understand the context.
            Base your answer ONLY on the provided context from the documents.
            You MUST cite sources from the new context using the format [C#].

            Conversation History:
            ---
            {formatted_history}---

            New Context from Documents:
            ---
            {truncated_context}
            ---
            
            User's New Query: "{query}"

            Synthesized Answer with Citations:
            """
            
            response = llm.invoke([HumanMessage(content=prompt)])
            final_answer = response.content
            logger.info("Successfully generated final answer with LLM.")
        except Exception as e:
            logger.exception(f"LLM generation failed: {e}")
            final_answer = "Sorry, an error occurred while generating the final answer."

        return {"answer": final_answer, "supporting_chunks": supporting_chunks}
            
            
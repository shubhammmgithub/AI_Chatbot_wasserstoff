import os
import logging
from typing import List, Dict, Any, Tuple, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from qdrant_client.http import models as qmodels

# Import the new Pydantic schema for structured reranker output
from backend.app.models.schemas import RerankResponse
from backend.app.core.config import get_qdrant_client, GROQ_MODEL, GROQ_API_KEY
from backend.app.services.embedding_service import EmbeddingService

logger = logging.getLogger("retrieval_service")

class RetrievalService:
    def __init__(self):
        self.qdrant_client = get_qdrant_client()
        self.embedding_service = EmbeddingService()
        
        # --- Final Answer LLM (large model) ---
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY, 
            model_name=GROQ_MODEL, 
            temperature=0.1
        ) 
        
        # --- Rerank LLM (small, faster model) ---
        # Get the rerank model name from .env, but fall back to the main model if not set
        rerank_model_name = os.getenv("RERANK_LLM_MODEL", GROQ_MODEL)
        rerank_llm_base = ChatGroq(
            api_key=GROQ_API_KEY, 
            model_name=rerank_model_name, 
            temperature=0.0
        )
        self.rerank_llm = rerank_llm_base.with_structured_output(RerankResponse)

    def _rerank_chunks_with_llm(
        self, query: str, hits: List[qmodels.ScoredPoint]
    ) -> Tuple[List[qmodels.ScoredPoint], List[float]]:
        """Reranks search hits using a smaller LLM as a relevance judge."""
        if not hits:
            return [], []
            
        logger.info(f"Reranking {len(hits)} hits using LLM...")

        # Format the chunks with their original indices for the prompt
        prompt_context = ""
        for i, hit in enumerate(hits):
            prompt_context += f"[DOCUMENT {i+1}]:\n{hit.payload.get('text', '')}\n\n"
            
        # Create a detailed prompt for the reranking task
        rerank_prompt = f"""
        You are an expert relevance judge. Your task is to analyze a list of retrieved document chunks 
        and re-order them based on how well they answer the user's query.
        
        User Query: "{query}"

        Here are the {len(hits)} document chunks to evaluate:
        ---
        {prompt_context}
        ---

        Please evaluate each document and return a sorted list of results, from most relevant to least relevant, 
        in the required JSON format.
        For each document, provide its original index (starting from 1), a relevance score from 0.0 to 1.0, 
        and a brief reason for the score.
        """
        
        try:
            # Call the structured LLM to get a validated Pydantic object
            response = self.rerank_llm.invoke(rerank_prompt)

            # --- CORRECTION: SIMPLIFIED AND MORE ROBUST SORTING LOGIC ---
            # Build a mapping of original index -> new score
            score_map = {result.index - 1: result.relevance_score for result in response.results}

            # Sort the original 'hits' list based on the new scores from the map
            # The lambda function looks up each hit's original index in the score_map
            sorted_hits = sorted(hits, key=lambda hit: score_map.get(hits.index(hit), 0.0), reverse=True)
            
            # Create a corresponding list of scores in the new correct order
            sorted_scores = [score_map.get(hits.index(hit), 0.0) for hit in sorted_hits]

            logger.info("Successfully reranked hits with LLM.")
            return sorted_hits, sorted_scores
            
        except Exception as e:
            logger.exception(f"LLM reranker failed: {e}. Falling back to original vector search order.")
            return hits, [h.score for h in hits]

    def retrieve_and_answer(
        self, query: str, session_id: str, top_k: int = 20, final_n: int = 5, chat_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Main pipeline: retrieve from Qdrant, rerank with LLM, and generate final answer."""
        collection_name = f"session_{session_id}"
        qvec = self.embedding_service._embed_texts([query])[0]
        
        try:
            hits = self.qdrant_client.search(
                collection_name=collection_name, 
                query_vector=qvec, 
                limit=top_k, 
                with_payload=True
            )
        except Exception:
            return {
                "answer": "Please ingest documents before asking questions.", 
                "supporting_chunks": []
            }
        
        if not hits:
            return {
                "answer": "I couldn't find anything relevant in your documents.", 
                "supporting_chunks": []
            }

        # --- Step 1: Rerank hits using LLM ---
        reranked_hits, rerank_scores = self._rerank_chunks_with_llm(query, hits)
        chosen_hits = reranked_hits[:final_n]
        
        # --- Step 2: Build history + context ---
        formatted_history = ""
        if chat_history:
            for message in chat_history:
                role = "User" if message['role'] == 'human' else "Assistant"
                formatted_history += f"{role}: {message['content']}\n"
        
        ctx_lines, supporting_chunks = [], []
        for i, h in enumerate(chosen_hits, start=1):
            p = h.payload
            entry = {
                "rank": i,
                "rerank_score": rerank_scores[i-1],
                "doc_id": p.get("doc_id"),
                "page": p.get("page"),
                "para": p.get("para"),
                "theme": p.get("theme"),
                "text": p.get("text"),
            }
            supporting_chunks.append(entry)
            ctx_lines.append(f"[C{i}] From '{entry['doc_id']}':\n{entry['text']}\n")
        context = "\n---\n".join(ctx_lines)
        
        # --- Step 3: Generate final answer with big LLM ---
        try:
            truncated_context = context[:7000]
            prompt = f"""
            You are an expert research assistant. Answer the user's new query based on the provided 
            conversation history and the new context retrieved from documents.
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
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            final_answer = response.content
        except Exception as e:
            logger.exception(f"LLM generation failed: {e}")
            final_answer = "Sorry, an error occurred while generating the final answer."

        return {"answer": final_answer, "supporting_chunks": supporting_chunks}
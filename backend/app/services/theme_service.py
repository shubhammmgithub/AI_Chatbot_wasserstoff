import logging
from typing import Dict, Any, Generator

# Pydantic is used to define our desired output structure
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from qdrant_client.http import models
from langchain_core.messages import HumanMessage

from backend.app.core.config import get_qdrant_client, GROQ_MODEL, GROQ_API_KEY

logger = logging.getLogger(__name__)

# --- 1. DEFINE THE DESIRED OUTPUT STRUCTURE ---
# We create a Pydantic class to tell the LLM exactly what format to return.
class ThemeAnalysis(BaseModel):
    """A structured analysis of a single theme from a document set."""
    theme_name: str = Field(description="A short, descriptive title of 3-5 words for the theme.")
    theme_summary: str = Field(description="A concise, 2-3 sentence summary of the main theme.")

class ThemeService:
    def __init__(self):
        self.qdrant_client = get_qdrant_client()
        
        # Initialize the base LLM
        llm = ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=0.1)
        
        # --- 2. CREATE THE STRUCTURED LLM ---
        # Chain the base LLM with the .with_structured_output() method, passing our desired schema
        self.structured_llm = llm.with_structured_output(ThemeAnalysis)

    def count_unique_themes(self, session_id: str) -> int:
        # This method remains the same and is correct.
        collection_name = f"session_{session_id}"
        try:
            all_points, _ = self.qdrant_client.scroll(
                collection_name=collection_name, limit=10000,
                with_payload=["theme"], with_vectors=False
            )
            if not all_points: return 0
            unique_labels = set(p.payload.get("theme") for p in all_points if p.payload.get("theme"))
            return len(unique_labels)
        except Exception:
            return 0

    def analyze_all_themes_stream(self, session_id: str) -> Generator[Dict[str, Any], None, None]:
        collection_name = f"session_{session_id}"
        logger.info(f"Starting structured analysis for collection: {collection_name}")
        try:
            # Code to get unique_theme_labels and themes_with_citations remains the same
            all_points, _ = self.qdrant_client.scroll(
                collection_name=collection_name, limit=10000,
                with_payload=["theme", "doc_id", "page", "para"]
            )
            themes_with_citations: Dict[str, list] = {}
            for point in all_points:
                theme_label = point.payload.get("theme")
                if theme_label:
                    themes_with_citations.setdefault(theme_label, []).append({
                        "doc_id": point.payload.get("doc_id"), "page": point.payload.get("page"),
                        "para": point.payload.get("para")
                    })
            unique_theme_labels = list(themes_with_citations.keys())
            
            for label in unique_theme_labels:
                # Code to get context_text is the same
                scroll_res, _ = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=models.Filter(must=[models.FieldCondition(key="theme", match=models.MatchValue(value=label))]),
                    limit=20, with_payload=True
                )
                context_text = "\n---\n".join([point.payload.get("text", "") for point in scroll_res])
                if not context_text: continue

                truncated_context = context_text[:7000]

                # --- 3. CREATE A SIMPLER PROMPT ---
                # We no longer need to tell the AI how to format JSON. We just ask it to do the analysis.
                analysis_prompt = f"""
                Analyze the following text excerpts and identify the main theme.
                Provide a descriptive name and a concise summary for this theme.

                Excerpts:
                ---
                {truncated_context}
                ---
                """
                
                try:
                    # --- 4. CALL THE STRUCTURED LLM ---
                    # The response is now a validated Pydantic object, not a raw string.
                    response = self.structured_llm.invoke(analysis_prompt)
                    
                    theme_name = response.theme_name
                    theme_summary = response.theme_summary

                except Exception as e:
                    logger.error(f"Structured LLM call failed for theme {label}: {e}")
                    theme_name = "Analysis Error"
                    theme_summary = "The AI model failed to generate a structured response."

                yield {
                    "name": theme_name, "summary": theme_summary,
                    "citations": themes_with_citations.get(label, []),
                    "original_label": label
                }
        except Exception as e:
            logger.exception(f"An error occurred during theme analysis stream for session {session_id}: {e}")
            yield {"error": "An error occurred during theme analysis."}
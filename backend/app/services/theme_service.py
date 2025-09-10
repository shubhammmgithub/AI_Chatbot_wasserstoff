import logging
from typing import Dict, Any, Generator
import json
import re
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from qdrant_client.http import models
from langchain_core.messages import HumanMessage

from backend.app.core.config import get_qdrant_client, GROQ_MODEL, GROQ_API_KEY

logger = logging.getLogger(__name__)

class ThemeAnalysis(BaseModel):
    """A structured analysis of a single theme from a document set."""
    theme_name: str = Field(description="A short, descriptive title of 3-5 words for the theme.")
    theme_summary: str = Field(description="A concise, 2-3 sentence summary of the main theme.")

class ThemeService:
    def __init__(self):
        self.qdrant_client = get_qdrant_client()
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=0.1)
        self.structured_llm = self.llm.with_structured_output(ThemeAnalysis)

    def count_unique_themes(self, session_id: str) -> int:
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
        
        # --- Code to get unique themes and citations (No changes here) ---
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
        
        # --- Main Analysis Loop ---
        for label in unique_theme_labels:
            theme_name = "Analysis Error"
            theme_summary = "Could not generate a valid response."
            
            try:
                # Code to get context_text is the same
                scroll_res, _ = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=models.Filter(must=[models.FieldCondition(key="theme", match=models.MatchValue(value=label))]),
                    limit=20, with_payload=True
                )
                context_text = "\n---\n".join([point.payload.get("text", "") for point in scroll_res])
                if not context_text: continue
                truncated_context = context_text[:7000]

                # --- PRIMARY ATTEMPT: Use the structured LLM ---
                analysis_prompt = f"Analyze the following text excerpts and provide a descriptive name and a concise summary for the main theme.\n\nExcerpts:\n---\n{truncated_context}\n---"
                response = self.structured_llm.invoke(analysis_prompt)
                theme_name = response.theme_name
                theme_summary = response.theme_summary

            except Exception as e:
                logger.warning(f"Structured output failed for theme {label}: {e}. Attempting fallback...")
                # --- FALLBACK LOGIC: If structured output fails, use manual parsing ---
                try:
                    fallback_prompt = f"""
                    Your task is to analyze the following text excerpts and generate a theme analysis.
                    You must respond with only a single, valid JSON object in the format:
                    {{"theme_name": "A short, descriptive title", "theme_summary": "A concise summary"}}
                    Do not add any explanation or markdown.

                    Excerpts: --- {truncated_context} ---
                    JSON Response:
                    """
                    response_content = self.llm.invoke([HumanMessage(content=fallback_prompt)]).content
                    
                    match = re.search(r'\{.*\}', response_content, re.DOTALL)
                    if match:
                        analysis = json.loads(match.group(0))
                        theme_name = analysis.get("theme_name", "Unnamed Theme")
                        theme_summary = analysis.get("theme_summary", "No summary.")
                    else:
                        theme_summary = "Fallback parsing also failed to find JSON."

                except Exception as fallback_e:
                    logger.error(f"Fallback parsing also failed for theme {label}: {fallback_e}")
            
            # --- FINAL YIELD ---
            # This is now correctly placed to yield a result for every theme in the loop.
            yield {
                "name": theme_name, "summary": theme_summary,
                "citations": themes_with_citations.get(label, []),
                "original_label": label
            }
import logging
import os
from typing import List, Tuple
import tqdm
import time
import google.generativeai as genai # type: ignore

# Check if Faiss is available to determine default device for CrossEncoder
FAISS_OK = False
try:
    import faiss # type: ignore
    FAISS_OK = True
except ImportError:
    pass # Keep FAISS_OK as False

MAX_RETRIES = 3
RETRY_DELAY = 5 # seconds

def rerank_gemini(pairs: List[Tuple[str, str]], api_key: str | None, model_name: str) -> List[float]:
    """Reranks candidate items based on query relevance using Gemini API.

    Args:
        pairs: List of (query_text, candidate_text) tuples.
        api_key: Gemini API key (can be None, will try env var).
        model_name: Name of the Gemini model to use.

    Returns:
        List of relevance scores (higher is better).
    """
    if not api_key:
        api_key = os.getenv("GAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("Gemini API key not provided and not found in environment variables (GAI_API_KEY or GEMINI_API_KEY).")
        return [0.0] * len(pairs) # Return default scores on error

    genai.configure(api_key=api_key)

    # Ensure the model name uses the expected prefix
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    logging.info(f"Using Gemini reranker model: {model_name}")
    model = genai.GenerativeModel(
        model_name,
        # Set safety settings to low to avoid blocking potentially relevant academic content
        safety_settings=[
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        )

    scores = []
    prompt_template = """Evaluate the relevance of the following document to the provided query. The query represents a user's research interests (based on their publication history). The document is a new research paper. Respond ONLY with a numerical score between 0.0 (not relevant) and 1.0 (highly relevant).

QUERY:
{query}

DOCUMENT:
{document}

RELEVANCE SCORE:"""

    logging.info(f"Reranking {len(pairs)} pairs using Gemini '{model_name}'...")
    for query, document in tqdm.tqdm(pairs, desc="Reranking with Gemini"):
        prompt = prompt_template.format(query=query, document=document)
        retries = 0
        while retries < MAX_RETRIES:
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        candidate_count=1,
                        temperature=0.0 # Low temperature for deterministic scoring
                    )
                )
                # Handle potential lack of candidates or text
                if response.candidates and response.candidates[0].content.parts:
                    score_text = response.candidates[0].content.parts[0].text.strip()
                    try:
                        score = float(score_text)
                        scores.append(max(0.0, min(1.0, score))) # Clamp score
                        break # Success, move to next pair
                    except ValueError:
                        logging.warning(f"Could not parse score from Gemini response: '{score_text}'. Assigning 0.0.")
                        scores.append(0.0)
                        break
                else:
                    logging.warning(f"Gemini response lacked valid content for a pair. Assigning 0.0. Response: {response}")
                    scores.append(0.0)
                    break
            except Exception as e:
                logging.warning(f"Gemini API call failed: {e}. Retrying ({retries+1}/{MAX_RETRIES})...")
                retries += 1
                if retries == MAX_RETRIES:
                    logging.error(f"Max retries reached for Gemini API call. Assigning 0.0.")
                    scores.append(0.0)
                else:
                    time.sleep(RETRY_DELAY * (2 ** retries)) # Exponential backoff
            except genai.types.BlockedPromptError as e:
                 logging.warning(f"Gemini API blocked prompt for safety reasons: {e}. Assigning 0.0.")
                 scores.append(0.0)
                 break # Don't retry if blocked
            except genai.types.StopCandidateException as e:
                 logging.warning(f"Gemini API stopped generation for safety reasons: {e}. Assigning 0.0.")
                 scores.append(0.0)
                 break # Don't retry if stopped

    logging.info("Gemini reranking complete.")
    return scores 
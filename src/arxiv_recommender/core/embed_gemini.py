import logging
import os
from typing import List
import time

import numpy as np
import tqdm

def embed_gemini(texts: List[str], api_key: str | None = None, model_name: str = "gemini-embedding-exp-03-07", batch: int = 32, task_type: str = "SEMANTIC_SIMILARITY") -> np.ndarray:
    """Embeds texts using the Google Gemini API.

    Args:
        texts: List of strings to embed.
        api_key: Google API Key. If None, tries to read from GAI_API_KEY environment variable.
        model_name: Name of the Gemini embedding model.
        batch: Batch size for API calls.
        task_type: The task type for the embedding (e.g., SEMANTIC_SIMILARITY, RETRIEVAL_DOCUMENT, CLUSTERING).

    Returns:
        Numpy array of embeddings.
    """
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        logging.error(
            "google-generativeai is not installed. Please install it: pip install google-generativeai"
        )
        raise

    resolved_api_key = api_key or os.environ.get("GAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("Gemini API key not provided and GAI_API_KEY environment variable not set.")

    genai.configure(api_key=resolved_api_key)
    logging.info(f"Using Gemini embedding model: {model_name}")

    vectors: List[List[float]] = []
    total_batches = (len(texts) + batch - 1) // batch
    logging.info(f"Encoding {len(texts)} texts in {total_batches} batches of size {batch} using Gemini API...")

    try:
        for i in tqdm.tqdm(range(0, len(texts), batch), total=total_batches, desc="Gemini Embedding"):
            max_retries = 5
            backoff_factor = 1.5
            # Filter out any empty or whitespace-only strings in the current batch
            chunk = [c for c in texts[i : i + batch] if c and c.strip()]
            if not chunk:
                continue
            # Retry logic for 429 errors
            res = None
            for attempt in range(1, max_retries + 1):
                try:
                    res = genai.embed_content(
                        model=model_name, content=chunk, task_type=task_type
                    )
                    break # Success, exit retry loop
                except Exception as e:
                    err_str = str(e).lower()
                    if "429" in err_str or "resource has been exhausted" in err_str:
                        wait = backoff_factor ** attempt
                        logging.warning(
                            f"Batch {i//batch+1}/{total_batches}: Quota exceeded (attempt {attempt}/{max_retries}). "
                            f"Retrying in {wait:.2f}s..."
                        )
                        time.sleep(wait)
                        continue
                    else:
                        logging.error(f"Batch {i//batch+1}/{total_batches}: Non-retryable error during Gemini API call: {e}")
                        raise # Re-raise other errors
            if res is None:
                # Exhausted retries
                logging.error(f"Batch {i//batch+1}/{total_batches}: Failed after {max_retries} retries due to rate limits.")
                raise RuntimeError("Failed to embed batch due to persistent rate limits.")

            # Extract embeddings (handle potential response variations)
            vecs = []
            if isinstance(res, dict) and "embeddings" in res:
                # Expected structure for batch embedding
                if isinstance(res["embeddings"], list):
                     vecs = [item["embedding"] for item in res["embeddings"] if "embedding" in item]
                else:
                    logging.warning(f"Batch {i//batch+1}/{total_batches}: Unexpected structure inside 'embeddings': {type(res['embeddings'])}. Skipping batch.")
            elif isinstance(res, dict) and "embedding" in res:
                # Handle case where response is {"embedding": [vec1, vec2, ...]}
                if isinstance(res["embedding"], list):
                    # Check if the items in the list are the vectors themselves
                    if res["embedding"] and isinstance(res["embedding"][0], list):
                        vecs = res["embedding"]
                    else:
                         logging.warning(f"Batch {i//batch+1}/{total_batches}: Items inside res['embedding'] are not lists (vectors). Skipping. Structure: {[type(item) for item in res['embedding'][:3]]}...")
                else:
                    logging.warning(f"Batch {i//batch+1}/{total_batches}: Unexpected structure for res['embedding']: {type(res['embedding'])}. Skipping batch.")
            elif isinstance(res, list):
                # Fallback: Maybe the response is directly a list of embeddings?
                if all(isinstance(item, list) for item in res):
                     vecs = res
                else:
                     logging.warning(f"Batch {i//batch+1}/{total_batches}: Response is a list, but items are not lists (embeddings). Skipping batch. Structure: {[type(item) for item in res[:3]]}...")
            else:
                # Fallback: If it's not a dict or list, log warning
                logging.warning(f"Batch {i//batch+1}/{total_batches}: Unexpected API response type: {type(res)}. Skipping batch.")

            if vecs:
                # Verify dimensions of the first vector in the batch
                first_vec_dim = len(vecs[0])
                logging.debug(f"Batch {i//batch+1}/{total_batches}: Received {len(vecs)} vectors, first vector dim: {first_vec_dim}")
            elif not chunk: # If chunk was empty, vecs will be empty, this is fine
                pass
            else: # Chunk was not empty, but vecs is empty - log this
                res_info = f"type={type(res)}"
                if isinstance(res, dict):
                    res_info += f", keys={list(res.keys())}"
                elif isinstance(res, list):
                    res_info += f", length={len(res)}, first_item_type={type(res[0]) if res else 'N/A'}"
                else:
                    res_info += f", value_preview={str(res)[:100]}" # Print first 100 chars for unknown types

                logging.warning(
                    f"Batch {i//batch+1}/{total_batches}: Embedding extraction resulted in empty list 'vecs'. "
                    f"API response info: {res_info}"
                )

            vectors.extend(vecs)
            # Add a small delay between successful requests
            time.sleep(0.05) # Minimal delay with high RPM limit
    except Exception as e:
        logging.error(f"Error during Gemini API call: {e}")
        # Consider more specific error handling based on genai library exceptions
        raise

    logging.info(f"Finished Gemini encoding. Received {len(vectors)} vectors.")
    if not vectors:
        # Return an empty array with the expected dimension if known, or handle appropriately
        # Placeholder: return empty 2D array. Requires knowing the dimension beforehand or handling downstream.
        logging.warning("Gemini embedding resulted in zero vectors.")
        # We need to know the expected dimension. Let's try to get it from the model configuration if possible,
        # otherwise raise an error or return a sentinel value.
        # For now, returning an empty array with shape (0, 0) might cause issues later.
        # Returning None or raising might be better.
        # Let's return an empty array of shape (0, D) if D is known, else raise.
        # For 'embedding-001', the dimension is 768
        dim = 768
        return np.empty((0, dim), dtype="float32")

    return np.asarray(vectors, dtype="float32") 
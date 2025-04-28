import json
import logging
import os
from pathlib import Path
from typing import Dict, List


def _ensure_cache(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


def _load_cache(path: Path) -> Dict[str, str]:
    _ensure_cache(path)
    with open(path, "r", encoding="utf-8") as fp:
        try:
            data = json.load(fp)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {}


def _save_cache(path: Path, data: Dict[str, str]):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def label_clusters_gemini(
    samples: Dict[int, List[str]],
    api_key: str | None = None,
    model_name: str = "gemini-1.5-pro-latest",
    cache_file: str | Path = "cache/cluster_labels.json",
) -> Dict[int, str]:
    """Generate human-readable labels for clusters using Gemini.

    Args:
        samples: Mapping of cluster_id -> list of sample texts.
        api_key: Gemini API key (fallback to env "GAI_API_KEY").
        model_name: Gemini model to use.
        cache_file: JSON file path to store previously generated labels.

    Returns:
        Dict of cluster_id -> label.
    """
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        raise ImportError("google-generativeai is required: pip install google-generativeai")

    resolved_api_key = api_key or os.environ.get("GAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("Gemini API key not provided and GAI_API_KEY not set.")

    genai.configure(api_key=resolved_api_key)
    model = genai.GenerativeModel(model_name)

    cache_path = Path(cache_file)
    cache = _load_cache(cache_path)

    labels: Dict[int, str] = {}
    prompt_tpl = (
        "The following paper titles and abstracts all belong to the same research field.\n"
        "Please concisely describe this field in English using no more than 10 words.\n"
        "Output only the field name on a single line."
    )

    for cid, texts in samples.items():
        str_cid = str(cid)
        cached_label = cache.get(str_cid)
        if cached_label and cached_label != "nan":
            labels[cid] = cached_label
            continue
        logging.info(f"Generating label for cluster {cid} (was: {cached_label})...")
        prompt = prompt_tpl + "\n---\n" + "\n\n".join(texts[:8])
        try:
            res = model.generate_content(prompt, generation_config={"temperature": 0.0})
            label = res.text.strip().split("\n")[0]
            if not label:
                logging.warning(f"Gemini returned an empty label for cluster {cid}. Marking as nan.")
                label = "nan"
        except Exception as e:
            logging.warning(f"Gemini labeling failed for cluster {cid}: {e}; using fallback label.")
            label = "nan"
        labels[cid] = label
        cache[str_cid] = label

    _save_cache(cache_path, cache)
    return labels 
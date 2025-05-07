#!/usr/bin/env python3
"""
ArXiv Recommender – modularized implementation
"""

from __future__ import annotations
import sys, pathlib
# Add package root (src/arxiv_recommender) to sys.path so that 'core' imports resolve
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

import argparse
import pathlib
import json
import datetime as dt
import logging
import numpy as np  # For saving/loading cached query embeddings
import re  # URLを検出するための正規表現
# from sklearn.cluster import KMeans # Removed as unused
from sklearn.metrics.pairwise import cosine_similarity # Add this import

from core.utils import load_config, setup_logger
from core.ingest import load_bibtex, fetch_arxiv
from core.embed_gemini import embed_gemini
from core.index import ANNIndex
from core.rerank import rerank_gemini
from core.cluster import cluster_embeddings, sample_texts_per_cluster
from core.label_gemini import label_clusters_gemini
# Add explanation support and helper imports
from core.explain import get_explanation
import hashlib
from tqdm import tqdm


def create_paper_text(paper: dict) -> str:
    """Creates a combined text representation (title, abstract, authors) for a paper dict."""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    
    authors_str = ""
    # Try to get authors from a list (common for arXiv parsed data)
    authors_list = paper.get("authors", []) 
    if isinstance(authors_list, list) and authors_list:
        authors_str = ", ".join(authors_list)
    else:
        # Fallback: try to get authors from a single string (common for BibTeX)
        # The key is often 'author' in BibTeX parsers
        bibtex_author_field = paper.get("author", "") 
        if isinstance(bibtex_author_field, str) and bibtex_author_field:
            # To make it somewhat consistent with the list joined by comma, 
            # we can split by 'and' and then join by ', '
            authors_parts = [a.strip() for a in bibtex_author_field.split(" and ")]
            authors_str = ", ".join(authors_parts)
        # If neither 'authors' (list) nor 'author' (string) yields content, authors_str remains ""

    text_parts = [title, abstract]
    if authors_str:
        text_parts.append(f"Authors: {authors_str}")
    # Join non-empty parts
    return "\n".join(filter(None, text_parts)).strip()


# コメント内のURLをリンクに変換する関数
def convert_urls_to_links(text):
    if not text:
        return ""
    # URLを検出する正規表現パターン
    url_pattern = r'(https?://[^\s<>"]+|www\.[^\s<>"]+)'
    
    def replace_url(match):
        url = match.group(0)
        # www.で始まるURLにはhttps://を追加
        href = url if url.startswith(('http://', 'https://')) else 'https://' + url
        return f'<a href="{href}" target="_blank" rel="noopener noreferrer">{url}</a>'
    
    # URLをリンクに変換
    return re.sub(url_pattern, replace_url, text)


def run(cfg: dict, bib_path: pathlib.Path, date: str | None, topk: int, refresh_embeddings: bool = False, cluster: bool = False, n_clusters: int = 3, debug: bool = False):
    # setup_logger()
    # Explicitly set log level to DEBUG
    setup_logger(level=logging.DEBUG)
    logging.info("Loading BibTeX…")
    my_papers = load_bibtex(bib_path)

    if debug:
        debug_sample_size = 100
        logging.warning(f"--- DEBUG MODE ACTIVE: Sampling first {debug_sample_size} BibTeX entries ---")
        if len(my_papers) > debug_sample_size:
             my_papers = my_papers[:debug_sample_size]

    query_texts = []
    valid_my_paper_indices = [] # Keep track of indices from my_papers that are actually used
    for idx, p in enumerate(my_papers):
        query_text = create_paper_text(p)
        if not query_text:
             logging.warning(f"Skipping BibTeX entry {p.get('id', 'N/A')} due to missing title, abstract, and authors.")
             continue
        query_texts.append(query_text)
        valid_my_paper_indices.append(idx) # Store the original index

    if not query_texts:
        logging.error("No valid query texts could be generated from the BibTeX file. Aborting.")
        return

    logging.info("Loading arXiv feed…")
    # Get categories from config, fallback to empty list (-> uses default CS_CATS in fetch_arxiv)
    fetch_cfg = cfg.get("fetch", {})
    arxiv_categories = fetch_cfg.get("arxiv_categories", [])
    if not isinstance(arxiv_categories, list): # Ensure it's a list
        logging.warning(f"'fetch.arxiv_categories' in config is not a list, using defaults.")
        arxiv_categories = [] # Use default if not a list

    # Pass categories to fetch_arxiv if the list is not empty
    if arxiv_categories:
         logging.info(f"Fetching arXiv papers for categories: {arxiv_categories}")
         arxiv_papers = fetch_arxiv(date, cats=arxiv_categories)
    else:
         # Use default categories defined in fetch_arxiv (CS_CATS)
         logging.info(f"No categories specified in config, using default CS categories.")
         arxiv_papers = fetch_arxiv(date)

    if not arxiv_papers:
        logging.error(f"No arXiv papers fetched for date {date}; aborting.")
        return

    # Create corpus texts including authors
    corpus_texts = []
    corpus_source_papers = [] # List of paper_dicts corresponding to corpus_texts/corpus_vecs
    for p_original in arxiv_papers: # Iterate over the original arxiv_papers
        corpus_text = create_paper_text(p_original)
        if not corpus_text:
             logging.warning(f"Skipping arXiv entry {p_original.get('id', 'N/A')} due to missing title, abstract, and authors.")
             continue 
        p_original['_corpus_index'] = len(corpus_texts) # Store index before adding to corpus_texts
        corpus_texts.append(corpus_text)
        corpus_source_papers.append(p_original) # Add the original paper dict that has text

    if not corpus_texts:
        logging.error("No valid corpus texts could be generated from the fetched arXiv papers. Aborting.")
        return

    # Always use Gemini provider
    provider = cfg.get("gemini", {})
    if not provider:
        logging.error("Missing 'gemini' configuration section in config file.")
        return

    # Always use Gemini embed function
    embed_fn = lambda texts, task="clustering": embed_gemini(
        texts,
        provider.get("api_key"),
        provider["model_name"],
        provider.get("batch_size", 32),
        task_type=task
    )

    # Embed query texts with caching
    cache_dir = pathlib.Path(cfg.get("cache_dir", "cache"))
    cache_dir.mkdir(exist_ok=True)
    # Include model name in cache file, replacing potentially problematic chars
    model_cache_key = provider.get("model_name", "unknown-model").replace("/", "-").replace(":", "-")
    # Add '_with_authors' suffix to distinguish cache files
    cache_file = cache_dir / f"{bib_path.stem}_{model_cache_key}_with_authors_query_vectors.npz"

    # --- Add Debug Logging for Cache ---
    logging.debug(f"Checking cache for query embeddings:")
    logging.debug(f"  refresh_embeddings flag: {refresh_embeddings}")
    logging.debug(f"  Expected cache file path: {cache_file.resolve()}") # resolve() で絶対パスを表示
    logging.debug(f"  Cache file exists check: {cache_file.exists()}")
    # --- End Debug Logging ---

    if refresh_embeddings or not cache_file.exists():
        logging.info(f"Embedding query texts (with authors) using Gemini and saving cache to {cache_file}")
        query_vecs = embed_fn(query_texts, task="retrieval_query") # Use retrieval_query for user profile
        np.savez(cache_file, query_vecs=query_vecs)
    else:
        logging.info(f"Loading cached query embeddings (with authors) from {cache_file}")
        data = np.load(cache_file)
        query_vecs = data["query_vecs"]

    # Check if query vectors are empty after loading/embedding
    if query_vecs.size == 0:
        logging.error("Failed to obtain valid query embeddings. Please check input data and API responses (run with LOGLEVEL=DEBUG). Aborting.")
        return

    # Embed corpus texts every run
    logging.info("Embedding corpus texts (with authors)…")
    corpus_vecs = embed_fn(corpus_texts, task="retrieval_document") # Use retrieval_document for candidates

    dim = query_vecs.shape[1]
    ann = ANNIndex(dim, cfg["index"]["factory"], cfg["index"].get("nprobe", 16))
    ann.train(corpus_vecs)
    ann.add(corpus_vecs)

    logging.info("ANN search…")
    oversample = cfg.get("rerank", {}).get("oversample", 4)
    cluster_centroids = []
    cluster_user_descs = []
    cluster_labels = None
    if cluster:
        # Auto clustering with UMAP + HDBSCAN
        logging.info("Running automatic clustering…")

        # --- Clustering Cache Logic --- 
        clustering_method_name = "hdbscan" # Hardcoded for now, could be from config
        cluster_cache_file = cache_dir / f"{cache_file.stem.replace('_query_vectors','')}_{clustering_method_name}_clusters.npz"

        if not refresh_embeddings and cluster_cache_file.exists():
            logging.info(f"Loading cached clustering results from {cluster_cache_file}")
            cluster_data = np.load(cluster_cache_file)
            cluster_labels = cluster_data["cluster_labels"]
            # Reconstruct centroids_dict
            centroids_dict = { 
                int(cid): vec for cid, vec in zip(cluster_data["centroid_ids"], cluster_data["centroid_vectors"])
            }
        else:
            logging.info(f"Running automatic clustering ({clustering_method_name}) on BibTeX embeddings (with authors)...")
            # TODO: Pass clustering params from config if needed
            cluster_labels, centroids_dict = cluster_embeddings(query_vecs)
            
            # Save clustering results to cache
            centroid_ids = np.array(list(centroids_dict.keys()), dtype=np.int64)
            centroid_vectors = np.array(list(centroids_dict.values()), dtype=np.float32)
            np.savez(
                cluster_cache_file,
                cluster_labels=cluster_labels,
                centroid_ids=centroid_ids,
                centroid_vectors=centroid_vectors
            )
            logging.info(f"Saved clustering results to {cluster_cache_file}")
        # --- End Clustering Cache Logic ---

        # Ensure cluster_labels and centroids_dict are available
        if cluster_labels is None or not centroids_dict:
             logging.error("Clustering failed or resulted in no clusters. Cannot proceed with cluster-based recommendations.")
             return # Exit if no clusters found

        # --- Get Cluster Labels (using cache) ---
        samples = sample_texts_per_cluster(cluster_labels, query_texts)
        # Human-readable labels via Gemini (skip if local mode to save API) – still allow
        try:
            # Pass the configured label model name from the cluster section
            label_model_name = cfg.get("cluster", {}).get("label_model", "gemini-1.5-pro-latest") # Fallback just in case
            # Use the label cache regardless of embedding/cluster cache
            label_cache_file = cache_dir / f"{bib_path.stem}_{clustering_method_name}_with_authors_labels.json"
            
            # If embeddings are refreshed (implying clustering is also refreshed),
            # we must also refresh the cluster labels by deleting the old label cache.
            if refresh_embeddings and label_cache_file.exists():
                logging.info(f"--refresh-embeddings is true, deleting stale cluster label cache: {label_cache_file}")
                try:
                    label_cache_file.unlink()
                except OSError as e:
                    logging.warning(f"Could not delete stale label cache {label_cache_file}: {e}")

            cluster_id_to_label = label_clusters_gemini(
                 samples, 
                 provider.get("api_key"), 
                 model_name=label_model_name,
                 # Consider making label cache name dependent on bib/model?
                 cache_file=label_cache_file 
            )
        except Exception as e:
            logging.warning(f"Cluster labeling failed: {e}; using numeric ids.")
            cluster_id_to_label = {cid: f"Cluster {cid}" for cid in centroids_dict.keys()}

        # --- Assign arXiv Papers to Closest Cluster --- 
        logging.info("Assigning arXiv papers to the closest cluster centroid...")
        num_arxiv_papers = corpus_vecs.shape[0]
        if num_arxiv_papers == 0:
            logging.warning("No arXiv papers to assign.")
            cluster_results = {cid: [] for cid in centroids_dict}
        else:
            # Create a temporary Faiss index for centroids
            centroid_vectors_list = list(centroids_dict.values())
            centroid_ids_list = list(centroids_dict.keys())
            # Use explicit "IDMap,FlatL2" factory string to avoid warning
            centroid_index = ANNIndex(dim=dim, factory="IDMap,FlatL2")
            centroid_index.add(np.array(centroid_vectors_list, dtype=np.float32),
                               ids=np.array(centroid_ids_list, dtype=np.int64))

            # Search for the nearest centroid for each arXiv paper
            distances, closest_centroid_ids = centroid_index.search(corpus_vecs, k=1)

            # Group papers by their assigned cluster
            papers_per_cluster: dict[int, list[tuple[int, float]]] = {cid: [] for cid in centroids_dict}
            for paper_idx in range(num_arxiv_papers):
                assigned_cid = int(closest_centroid_ids[paper_idx][0])
                distance = distances[paper_idx][0]
                if assigned_cid != -1: # Check if assignment was successful
                    papers_per_cluster[assigned_cid].append((paper_idx, distance))

            # --- Rank and Rerank within each cluster ---
            cluster_results = {}
            # Get user descriptions prepared earlier
            cluster_user_descs_dict = {cid: "\n---\n".join(samples.get(cid, [])) for cid in centroids_dict.keys()}

            for cid, assigned_papers in papers_per_cluster.items():
                if not assigned_papers:
                    cluster_results[cid] = []
                    continue

                # Sort assigned papers by distance to centroid (ascending)
                sorted_papers = sorted(assigned_papers, key=lambda x: x[1])
                
                # Get top K candidates based on distance
                initial_candidates_indices = [idx for idx, dist in sorted_papers[:topk * oversample]]
                # Ensure indices are valid before accessing corpus_source_papers and corpus_texts
                valid_indices = [i for i in initial_candidates_indices if i < len(corpus_source_papers)]
                initial_candidates = [corpus_source_papers[i] for i in valid_indices] # Use corpus_source_papers
                candidate_texts = [corpus_texts[i] for i in valid_indices] # Get corresponding texts from corpus_texts

                if cfg["rerank"]["enable"]:
                    logging.info(f"Reranking top {len(initial_candidates)} candidates for cluster {cid}...")
                    user_desc = cluster_user_descs_dict.get(cid, "")
                    pairs = [(user_desc, cand_text) for cand_text in candidate_texts]
                    if not pairs:
                         logging.warning(f"No valid pairs generated for reranking in cluster {cid}. Skipping reranking.")
                         # Ensure idx is valid for corpus_source_papers
                         scored = [(corpus_source_papers[idx], float(-dist)) for idx, dist in sorted_papers[:topk] if idx < len(corpus_source_papers)]
                         ranked = scored
                    else:
                         scores = rerank_gemini(pairs, provider.get("api_key"), cfg["rerank"].get("gemini_model", "gemini-1.5-pro-latest"))
                         # Need to align scores back to original candidates
                         ranked = sorted(zip(initial_candidates, scores), key=lambda x: x[1], reverse=True)[:topk]
                else:
                    # Use distance-based ranking if rerank is disabled
                    # Score is negative distance (higher is better)
                    # Ensure idx is valid for corpus_source_papers
                    scored = [(corpus_source_papers[idx], float(-dist)) for idx, dist in sorted_papers[:topk] if idx < len(corpus_source_papers)]
                    ranked = scored # Already sorted by distance implicitly if ANN Index returns sorted

                cluster_results[cid] = ranked

    else: # Not clustering
        # --- Original Non-clustering Path --- 
        logging.info("Running non-clustered recommendation...")
        centroid_vec = query_vecs.mean(axis=0, keepdims=True)
        user_desc = "\n---\n".join(query_texts) # Use the already generated query texts
        distances, idxs = ann.search(centroid_vec, k=topk * oversample)
        # Ensure idxs is not empty and has the expected structure
        if idxs.size == 0:
            logging.warning("ANN search returned no results.")
            candidates = []
            initial_scores = []
            candidate_texts_for_rerank = [] # Initialize for reranking
        else:
            candidate_indices = idxs[0].tolist()
            initial_scores = distances[0].tolist()
            # Ensure indices are valid for corpus_source_papers
            valid_candidate_indices = [i for i in candidate_indices if i < len(corpus_source_papers)]
            candidates = [corpus_source_papers[i] for i in valid_candidate_indices]
            candidate_texts_for_rerank = [corpus_texts[i] for i in valid_candidate_indices]

        if cfg["rerank"]["enable"]:
            pairs = [(user_desc, cand_text) for cand_text in candidate_texts_for_rerank[:len(candidates)]]
            # Always use Gemini rerank
            scores = rerank_gemini(pairs, provider.get("api_key"), cfg["rerank"].get("gemini_model", "gemini-1.5-pro-latest"))
            # Rerank only provides scores, limit to topk after sorting
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:topk]
        else:
            scored = [(candidates[i], float(-initial_scores[i])) for i in range(min(topk, len(candidates)))]
            ranked = scored # Already sorted by distance implicitly if ANN Index returns sorted
        # Store result for the single non-cluster case
        cluster_results = {0: ranked} # Use 0 as the pseudo-cluster id

    # --- Identify related user papers for each recommendation ---
    logging.info("Identifying related user papers for each recommendation...")
    num_related_to_show = 3  # Number of related user papers to show

    for cid in cluster_results:
        updated_ranked_list_for_cluster = []
        for paper_dict, score in cluster_results[cid]:
            # Ensure _corpus_index exists, indicating it's from corpus_source_papers
            if '_corpus_index' not in paper_dict:
                logging.warning(f"Paper ID {paper_dict.get('id', 'N/A')} missing '_corpus_index'. Skipping related user paper identification.")
                paper_dict['related_user_papers'] = []
                updated_ranked_list_for_cluster.append((paper_dict, score))
                continue

            arxiv_paper_corpus_index = paper_dict['_corpus_index']
            
            # corpus_vecs and query_vecs checks
            if corpus_vecs is None or arxiv_paper_corpus_index >= len(corpus_vecs):
                logging.warning(f"Corpus vectors not available or index out of bounds for paper ID {paper_dict.get('id', 'N/A')}. Skipping.")
                paper_dict['related_user_papers'] = []
                updated_ranked_list_for_cluster.append((paper_dict, score))
                continue
            
            recommended_arxiv_vec = corpus_vecs[arxiv_paper_corpus_index].reshape(1, -1)

            if query_vecs is None or query_vecs.size == 0:
                logging.warning("Query vectors are empty. Skipping related user paper identification for all recommendations.")
                # Add empty list to all remaining papers in this cluster if query_vecs is missing
                paper_dict['related_user_papers'] = []
                updated_ranked_list_for_cluster.append((paper_dict, score))
                # We can break the inner loop or continue, but subsequent papers will also fail here.
                # For safety, just mark this one and let others be processed (though they'll also hit this).
                # A more global flag could be set if query_vecs is globally problematic.
                continue 

            similarities = cosine_similarity(recommended_arxiv_vec, query_vecs)[0]
            
            related_papers_with_scores = []
            for i, similarity_score in enumerate(similarities):
                if i < len(valid_my_paper_indices): # Ensure i is a valid index for valid_my_paper_indices
                    original_my_paper_idx = valid_my_paper_indices[i]
                    user_paper = my_papers[original_my_paper_idx] # Get the correct user paper
                    
                    # Construct link for user paper
                    user_paper_link = None
                    if user_paper.get('doi'):
                        user_paper_link = f"https://doi.org/{user_paper.get('doi')}"
                    elif user_paper.get('eprint'):
                        user_paper_link = f"https://arxiv.org/abs/{user_paper.get('eprint')}"
                    elif user_paper.get('url'): # Fallback to general URL if available
                        user_paper_link = user_paper.get('url')
                        
                    related_papers_with_scores.append({
                        'title': user_paper.get('title', 'N/A'),
                        'id': user_paper.get('ID', user_paper.get('id', 'N/A')), # BibTeX key is often 'ID'
                        'similarity': float(similarity_score),
                        'author_display_string': user_paper.get('author', 'N/A'), # Use new key, assuming 'author' holds the raw string
                        'link': user_paper_link
                    })
            
            sorted_related_papers = sorted(related_papers_with_scores, key=lambda x: x['similarity'], reverse=True)
            paper_dict['related_user_papers'] = sorted_related_papers[:num_related_to_show]
            updated_ranked_list_for_cluster.append((paper_dict, score))
        cluster_results[cid] = updated_ranked_list_for_cluster
    # --- End of identifying related user papers ---

    # Prepare explanation configuration for HTML
    exp_cfg = cfg.get('explanation', {})
    exp_enabled = exp_cfg.get('enable', False)
    # Compute consistent hash for caching
    exp_config_hash = hashlib.md5(f"{exp_cfg.get('model_name','')}-{exp_cfg.get('target_language','')}.encode()".encode()).hexdigest()[:8]
    exp_cache_subdir = exp_cfg.get('cache_dir', 'explanation_cache')
    # Precompute explanations if enabled
    if exp_enabled and exp_cfg.get('precompute', False):
        logging.info("Precomputing explanations for recommended papers...")
        for cid, ranked_list in tqdm(cluster_results.items()):
            for paper, _ in ranked_list:
                # Check if paper is valid dict with 'id'
                if isinstance(paper, dict) and 'id' in paper:
                     get_explanation(paper['id'], cfg)
                else:
                     logging.warning(f"Skipping precomputation for invalid paper data: {paper}")

    out_cfg = cfg.get("output", {})
    out_dir = pathlib.Path(out_cfg.get("dir", "output"))
    out_dir.mkdir(exist_ok=True)
    # Determine the date string for the filename
    if date:
        try:
            # Convert 'YYYY-MM-DD' to 'YYYYMMDD'
            file_date_str = dt.datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")
        except ValueError:
            logging.warning(f"Invalid date format '{date}'. Using current UTC date for filename.")
            file_date_str = dt.datetime.utcnow().strftime("%Y%m%d")
    else:
        # Use current UTC date if no date argument is provided
        file_date_str = dt.datetime.utcnow().strftime("%Y%m%d")

    json_path = out_dir / f"arxiv_reco_{file_date_str}.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        if cluster:
            json.dump(cluster_results, fp, ensure_ascii=False, indent=2)
        else:
            json.dump(ranked, fp, ensure_ascii=False, indent=2)

    html_path = out_dir / f"arxiv_reco_{file_date_str}.html"
    with open(html_path, "w", encoding="utf-8") as fp:
        # Add DOCTYPE, head with charset and Marked.js CDN
        fp.write("<!DOCTYPE html>")
        fp.write("<html><head>")
        fp.write("    <meta charset=\"UTF-8\">")
        fp.write("    <title>arXiv Recommendations</title>")
        # Include Marked.js library from CDN
        fp.write("    <script src=\"https://cdn.jsdelivr.net/npm/marked/marked.min.js\"></script>")
        # Include MathJax library from CDN for LaTeX math rendering
        fp.write("    <script src=\"https://polyfill.io/v3/polyfill.min.js?features=es6\"></script>")
        fp.write("    <script id=\"MathJax-script\" async src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js\"></script>")
        fp.write("    <script>")
        fp.write("    window.MathJax = {")
        fp.write("        tex: {")
        fp.write("            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],")
        fp.write("            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],")
        fp.write("            processEscapes: true,")
        fp.write("            processEnvironments: true")
        fp.write("        },")
        fp.write("        options: {")
        fp.write("            ignoreHtmlClass: 'ignore-mathjax',")
        fp.write("            processHtmlClass: 'process-mathjax'")
        fp.write("        }")
        fp.write("    };")
        fp.write("    </script>")
        fp.write("    <style>")
        fp.write("        .explanation { display:none; margin-top:0.5em; padding:0.5em; border:1px solid #ddd; background:#f9f9f9; position: relative; }")
        fp.write("        .hide-btn { position: absolute; top: 5px; right: 5px; cursor: pointer; font-size: 0.8em; padding: 2px 5px; border: 1px solid #ccc; background: #eee; border-radius: 3px; }")
        fp.write("        .content { margin-top: 5px; } /* Add some space below hide button */") # Added margin to content
        fp.write("        .comments { color: #0066cc; font-style: italic; margin-left: 5px; }") # コメント表示のスタイル
        fp.write("        .comments-label { font-weight: bold; color: #555; margin-left: 10px; }") # コメントラベルのスタイル
        fp.write("        .comments a { color: #0066dd; text-decoration: underline; font-weight: bold; }") # コメント内リンクのスタイル
        fp.write("        .comments a:hover { color: #004499; text-decoration: underline; }") # リンクのホバー時スタイル
        fp.write("    </style>")
        fp.write("</head><body>") # Close head, open body

        if cluster:
            sorted_cids = sorted(cluster_results.keys()) if cluster_results else []
            for cid in sorted_cids:
                ranked = cluster_results[cid]
                label = cluster_id_to_label.get(cid, f"Cluster {cid} (ID: {cid})")
                fp.write(f"<h2>{label}</h2><ol>")
                for paper, score in ranked:
                    sanitized_id = paper["id"].replace('.', '-').replace('/', '_')
                    authors_str = ", ".join(paper.get("authors", []))
                    category = paper.get("primary_category", "")
                    comments = paper.get("comments")
                    comments_html = f' <span class="comments-label">Comments:</span> <span class="comments">{convert_urls_to_links(comments)}</span>' if comments else ''
                    
                    # Prepare related papers entries
                    related_papers_entries = []
                    for rel_paper in paper.get('related_user_papers', []):
                        link_start = f'<a href="{rel_paper["link"]}" target="_blank">' if rel_paper.get("link") else ""
                        link_end = "</a>" if rel_paper.get("link") else ""
                        rel_authors_raw_str = rel_paper.get('author_display_string', 'N/A')
                        rel_authors_display = 'N/A'
                        if rel_authors_raw_str != 'N/A' and isinstance(rel_authors_raw_str, str):
                            authors_parts = [a.strip() for a in rel_authors_raw_str.split('and')]
                            rel_authors_display = " and ".join(authors_parts)
                        elif isinstance(rel_authors_raw_str, list):
                            rel_authors_display = ", ".join(rel_authors_raw_str)
                        related_papers_entries.append(
                            f"<li>{link_start}{rel_paper['title']}{link_end} (<i>{rel_authors_display}</i>) - Score: {rel_paper['similarity']:.3f}</li>"
                        )

                    # Main paper line
                    fp.write(
                        f'<li>'
                        f'<a href="https://arxiv.org/abs/{paper["id"]}" target="_blank">{paper["title"]}</a><br>'
                        f'<small><i>{authors_str}</i> [{category}]{comments_html}</small> – score {score:.3f} '
                    )
                    # Explain button (if enabled)
                    if exp_enabled:
                        fp.write(f'<button class="explain-btn" data-id="{paper["id"]}" data-target-div="explain-{sanitized_id}">Explain</button>')
                    
                    # Show related papers using <details>
                    if related_papers_entries:
                        fp.write('<details style="margin-top:0.5em; margin-left:15px;">')
                        fp.write('<summary>Most related from your library</summary>')
                        fp.write('<ul>')
                        for entry in related_papers_entries:
                            fp.write(entry)
                        fp.write('</ul>')
                        fp.write('</details>')

                    # Explanation div (if enabled)
                    if exp_enabled:
                        # Explanation container for cluster item
                        fp.write('<div class="explanation" id="explain-{}">'.format(sanitized_id))
                        fp.write('<button class="hide-btn" onclick="document.getElementById(\'explain-{}\').style.display=\'none\';">Hide</button>'.format(sanitized_id))
                        fp.write('<div class="content"></div>')
                        fp.write('</div>')
                    
                    fp.write(f'</li>') # Close list item
                fp.write("</ol>")
        else: # Non-clustered case
            fp.write("<h2>Top Recommendations</h2><ol>")
            for paper, score in cluster_results.get(0, []):
                sanitized_id = paper["id"].replace('.', '-').replace('/', '_')
                authors_str = ", ".join(paper.get("authors", []))
                category = paper.get("primary_category", "")
                comments = paper.get("comments")
                comments_html = f' <span class="comments-label">Comments:</span> <span class="comments">{convert_urls_to_links(comments)}</span>' if comments else ''

                # Prepare related papers entries
                related_papers_entries = []
                for rel_paper in paper.get('related_user_papers', []):
                    link_start = f'<a href="{rel_paper["link"]}" target="_blank">' if rel_paper.get("link") else ""
                    link_end = "</a>" if rel_paper.get("link") else ""
                    rel_authors_raw_str = rel_paper.get('author_display_string', 'N/A')
                    rel_authors_display = 'N/A'
                    if rel_authors_raw_str != 'N/A' and isinstance(rel_authors_raw_str, str):
                        authors_parts = [a.strip() for a in rel_authors_raw_str.split('and')]
                        rel_authors_display = " and ".join(authors_parts)
                    elif isinstance(rel_authors_raw_str, list):
                        rel_authors_display = ", ".join(rel_authors_raw_str)
                    related_papers_entries.append(
                        f"<li>{link_start}{rel_paper['title']}{link_end} (<i>{rel_authors_display}</i>) - Score: {rel_paper['similarity']:.3f}</li>"
                    )

                # Main paper line
                fp.write(
                    f'<li>'
                    f'<a href="https://arxiv.org/abs/{paper["id"]}" target="_blank">{paper["title"]}</a><br>'
                    f'<small><i>{authors_str}</i> [{category}]{comments_html}</small> – score {score:.3f} '
                )
                # Explain button (if enabled)
                if exp_enabled:
                    fp.write(f'<button class="explain-btn" data-id="{paper["id"]}" data-target-div="explain-{sanitized_id}">Explain</button>')
                
                # Show related papers using <details>
                if related_papers_entries:
                    fp.write('<details style="margin-top:0.5em; margin-left:15px;">')
                    fp.write('<summary>Most related from your library</summary>')
                    fp.write('<ul>')
                    for entry in related_papers_entries:
                        fp.write(entry)
                    fp.write('</ul>')
                    fp.write('</details>')

                # Explanation div (if enabled)
                if exp_enabled:
                    # Explanation container for non-cluster item
                    fp.write('<div class="explanation" id="explain-{}">'.format(sanitized_id))
                    fp.write('<button class="hide-btn" onclick="document.getElementById(\'explain-{}\').style.display=\'none\';">Hide</button>'.format(sanitized_id))
                    fp.write('<div class="content"></div>')
                    fp.write('</div>')
                
                fp.write(f'</li>') # Close list item
            fp.write("</ol>")

        # Close body tag before adding scripts
        fp.write("</body>")
        # Add explanation interaction JavaScript if enabled
        if exp_enabled: # This condition now covers both explain and related papers buttons if exp_enabled is a general proxy for JS interactivity
            fp.write(f"""
<script>
const expEnabled = {str(exp_enabled).lower()};
const expConfigHash = '{exp_config_hash}';
const proxyBaseUrl = 'http://localhost:5001'; // Assuming default proxy port

document.querySelectorAll('.explain-btn').forEach(btn => {{
    btn.addEventListener('click', async () => {{
        const pid = btn.dataset.id;
        const targetDivId = btn.dataset.targetDiv;
        const div = document.getElementById(targetDivId);
        const contentDiv = div.querySelector('.content');
        const button = btn;

        div.style.display = 'block';
        contentDiv.innerHTML = '<p>Generating...</p>';
        button.disabled = true;
        button.innerText = 'Generating...';

        const storageKey = `explanation_${{pid}}_${{expConfigHash}}`;
        let explanation = localStorage.getItem(storageKey);

        if (explanation) {{
            console.log("Using cached explanation for", pid);
            try {{
                const cleanedExplanation = explanation.replace(/^```markdown\\n/, '').replace(/\\n```$/, '');
                contentDiv.innerHTML = marked.parse(cleanedExplanation, {{ sanitize: true }});
                if (window.MathJax) {{
                    MathJax.typesetPromise([contentDiv]).catch((err) => console.error('MathJax error:', err));
                }}
            }} catch (e) {{
                console.error('Error parsing cached markdown:', e);
                contentDiv.innerHTML = '<p>Error rendering cached explanation.</p>';
            }}
            button.innerText = 'Explain';
            button.disabled = false;
            return;
        }}

        try {{
            contentDiv.innerHTML = '<p>Fetching explanation from local proxy...</p>';
            const resp = await fetch(`${{proxyBaseUrl}}/explain?id=${{pid}}`);

            if (resp.ok) {{
                const data = await resp.json();
                if (data.explanation) {{
                    explanation = data.explanation;
                    console.log("Received explanation for", pid);
                    try {{
                        const cleanedExplanation = explanation.replace(/^```markdown\\n/, '').replace(/\\n```$/, '');
                        contentDiv.innerHTML = marked.parse(cleanedExplanation, {{ sanitize: true }});
                        if (window.MathJax) {{
                            MathJax.typesetPromise([contentDiv]).catch((err) => console.error('MathJax error:', err));
                        }}
                    }} catch (e) {{
                         console.error('Error parsing received markdown:', e);
                         contentDiv.innerHTML = '<p>Error rendering explanation.</p>';
                    }}
                    localStorage.setItem(storageKey, explanation);
                }} else {{
                    contentDiv.innerHTML = '<p>Error from proxy: ' + (data.error || 'Unknown error') + '</p>';
                    localStorage.removeItem(storageKey);
                }}
            }} else {{
                contentDiv.innerHTML = `<p>Error fetching from proxy: ${{resp.status}} ${{resp.statusText}}. Ensure the proxy server is running (python src/arxiv_recommender/server/proxy.py).</p>`;
                localStorage.removeItem(storageKey);
            }}
        }} catch (err) {{
            contentDiv.innerHTML = '<p>Network error fetching explanation: ' + err + '. Is the proxy server running on ' + proxyBaseUrl + '?</p>';
            localStorage.removeItem(storageKey);
        }} finally {{
             button.innerText = 'Explain';
             button.disabled = false;
        }}
    }});
}});
</script>
""")

        # Close html tag
        fp.write("</html>")

    logging.info(f"Done. Results saved to {json_path} & {html_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="arXiv recommender")
    ap.add_argument("--explain-id", type=str, default=None, help="arXiv ID to generate explanation and exit")
    ap.add_argument("--config", type=pathlib.Path, default="config.yaml")
    ap.add_argument("--bib", type=pathlib.Path, required=True, help="Paperpile .bib file path")
    ap.add_argument("--debug", action="store_true", help="Run in debug mode with sampled BibTeX data (first 100 entries)")
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (default today UTC)")
    ap.add_argument("--topk", type=int, default=None, help="Number of top recommendations (overrides config.output.top_k)")
    ap.add_argument("--refresh-embeddings", action="store_true", help="Recompute and cache BibTeX query embeddings")
    ap.add_argument("--cluster", action="store_true", help="Enable clustering of BibTeX entries")
    ap.add_argument("--n_clusters", type=int, default=3, help="Number of clusters when clustering is enabled")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    # CLI explain mode: generate explanation for a single arXiv ID and exit
    if args.explain_id:
        explanation = get_explanation(args.explain_id, cfg)
        if explanation:
            print(explanation)
        else:
            print(f"Failed to get explanation for {args.explain_id}", file=sys.stderr)
        return
    # Run recommendation pipeline
    # Mode is always gemini
    topk = args.topk or cfg.get("output", {}).get("top_k", 50)
    run(cfg, args.bib, args.date, topk, args.refresh_embeddings, args.cluster, args.n_clusters, args.debug)


if __name__ == "__main__":
    main()

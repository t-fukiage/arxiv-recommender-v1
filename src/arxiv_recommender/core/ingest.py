from __future__ import annotations

import datetime as dt
import logging
import pathlib
import re # 正規表現ライブラリを追加
from typing import List, Dict, Any

import bibtexparser # type: ignore
# import feedparser # type: ignore

import arxiv, logging, urllib.parse

CS_CATS = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "cs.RO"]

# --- Helper function to remove simple LaTeX commands ---
def remove_latex_commands(text: str | Any) -> str:
    """Removes some common problematic LaTeX commands like \\footnote."""
    if not isinstance(text, str):
        return "" # Return empty string if not a string

    # Remove \\footnote{...}
    text = re.sub(r'\\footnote\{.*?\}', '', text, flags=re.DOTALL)
    # Remove common inline commands like \\textbf{...} -> ...
    text = re.sub(r'\\textbf\{(.*?)\}', r'\\1', text, flags=re.DOTALL)
    text = re.sub(r'\\textit\{(.*?)\}', r'\\1', text, flags=re.DOTALL)
    text = re.sub(r'\\texttt\{(.*?)\}', r'\\1', text, flags=re.DOTALL)
    # Add more rules as needed, e.g., for math environments if they cause issues
    # Consider removing remaining single backslashes if they are not part of valid LaTeX for the parser
    # text = text.replace('\\\\', '') # Be careful with this, might remove intended characters

    # Remove potential leftover curly braces around the whole string if any
    text = text.strip('{}')
    return text.strip() # Remove leading/trailing whitespace

# --- Updated load_bibtex function ---
def load_bibtex(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Loads and parses a BibTeX file using bibtexparser v2.

    Args:
        path: Path to the .bib file.

    Returns:
        List of dictionaries, each representing a paper with
        'title', 'abstract', 'id', 'year', etc.
    """
    logging.info(f"Loading BibTeX file from: {path} (using bibtexparser v2)")
    bib_database = None
    try:
        bib_database = bibtexparser.parse_file(path)
        logging.info(f"BibTeX parsing finished. Type of bib_database: {type(bib_database)}")
    except FileNotFoundError:
        logging.error(f"BibTeX file not found at: {path}")
        return []
    except Exception as e:
        logging.error(f"Error parsing BibTeX file with bibtexparser v2: {e}")
        if bib_database is None or not hasattr(bib_database, 'entries') or not bib_database.entries:
             logging.error("Parsing failed completely or resulted in no entries.")
             return []
        else:
             logging.warning("Proceeding with potentially partially parsed BibTeX data due to error.")

    entries = []
    if hasattr(bib_database, 'entries'):
        num_raw_entries = len(bib_database.entries)
        logging.info(f"Found {num_raw_entries} raw entries in bib_database.")
        if num_raw_entries > 0:
             # Log details about the first entry for inspection
             first_entry = bib_database.entries[0]
             logging.debug(f"Type of first entry: {type(first_entry)}")
             # Assuming entry is dict-like, log its keys
             if isinstance(first_entry, dict):
                  logging.debug(f"Keys of first entry: {list(first_entry.keys())}")

        processed_count = 0
        skipped_count = 0
        for i, entry in enumerate(bib_database.entries):
            # Use entry.key as the citation key (ID)
            entry_id_for_log = getattr(entry, 'key', f'entry_index_{i}')
            try:
                # Extract title from fields_dict
                title_field = entry.fields_dict.get('title')
                original_title = title_field.value if title_field and title_field.value else ''
                # Extract abstract with fallback to annote
                abstract_field = entry.fields_dict.get('abstract') or entry.fields_dict.get('annote')
                original_abstract = abstract_field.value if abstract_field and abstract_field.value else ''

                # Log the extracted original title
                logging.debug(f"Processing {entry_id_for_log}: Original Title = '{original_title}'")

                # Apply LaTeX command removal
                # title = remove_latex_commands(original_title)
                # abstract = remove_latex_commands(original_abstract)
                title = original_title
                abstract = original_abstract

                logging.debug(f"Processing {entry_id_for_log}: Cleaned Title = '{title}'")

                # Skip if title is empty
                if not title:
                    logging.warning(f"Skipping entry {entry_id_for_log}: empty title after cleaning.")
                    skipped_count += 1
                    continue

                # Append cleaned entry
                entries.append({
                    "title": title,
                    "abstract": abstract,
                    "id": entry_id_for_log,
                    "arxiv_id": entry['eprint'] if 'eprint' in entry.fields_dict else None,
                    "doi": entry['doi'] if 'doi' in entry.fields_dict else None,
                    "url": entry['url'] if 'url' in entry.fields_dict else None,
                    "year": entry['year'] if 'year' in entry.fields_dict else None,
                    "entry_type": entry.entry_type,
                })
                processed_count += 1
            except Exception as inner_e:
                logging.warning(f"Failed to process entry {entry_id_for_log} due to: {inner_e}", exc_info=True)
                skipped_count += 1
                continue
        logging.info(f"Entry processing summary: Processed={processed_count}, Skipped={skipped_count}")
    else:
         logging.error("Parsed BibDatabase object does not contain 'entries'.")

    logging.info(f"Successfully processed {len(entries)} entries from BibTeX (using bibtexparser v2).")
    return entries

def fetch_arxiv(date: str | None = None,
                    cats: list[str] = CS_CATS,
                    max_results: int = 2000):
    """
    指定日の CS 系論文を arxiv.py で取得する（日付は GMT 基準）。
    """
    # Handle 'today' as input for date
    if date is None or date.lower() == 'today':
        # Use today's UTC date
        date_obj = dt.datetime.utcnow().date()
        logging.info(f"Fetching arXiv papers for today (UTC): {date_obj.isoformat()}")
    else:
        try:
            # Parse the provided date string
            date_obj = dt.date.fromisoformat(date)
            logging.info(f"Fetching arXiv papers for specified date (UTC): {date_obj.isoformat()}")
        except ValueError:
            logging.error(f"Invalid date format: '{date}'. Please use YYYY-MM-DD or 'today'. Using today's date instead.")
            date_obj = dt.datetime.utcnow().date() # Fallback to today

    date_str = date_obj.strftime("%Y%m%d")
    date_range = f"submittedDate:[{date_str}0000 TO {date_str}2359]"

    cat_query = "(" + " OR ".join(f"cat:{c}" for c in cats) + ")"
    query = f"{cat_query} AND {date_range}"

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    papers = [{
        "id": r.entry_id.rsplit("/", 1)[-1],
        "title": r.title,
        "abstract": r.summary.replace("\n", " "),
        "published": r.published.isoformat(),
        "url": r.pdf_url,
        "authors": [str(author) for author in r.authors],
        "primary_category": r.primary_category,
    } for r in client.results(search)]
    logging.info("Fetched %d papers for %s (UTC)", len(papers), date_obj)
    return papers

# def fetch_arxiv(date: str | None = None) -> List[Dict[str, str]]:
#     """Fetches arXiv papers for a specific date via RSS feed.

#     Args:
#         date: Date string in YYYY-MM-DD format. Defaults to today (UTC).

#     Returns:
#         List of dictionaries, each representing an arXiv paper with
#         'title', 'abstract', 'id'.
#     """
#     if date is None:
#         date = dt.datetime.utcnow().strftime("%Y-%m-%d")
#         logging.info(f"Fetching arXiv papers for today: {date}")
#     else:
#         logging.info(f"Fetching arXiv papers for date: {date}")

#     try:
#         y, m, d = map(int, date.split("-"))
#         date_obj = dt.date(y, m, d)
#     except ValueError:
#         logging.error(f"Invalid date format: {date}. Please use YYYY-MM-DD.")
#         return []

#     # Use date-range query to ensure no missing papers; fetch up to 2000 results in one call
#     # Build CS subcategory OR-list
#     cs_cats = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "cs.RO"]
#     cat_query = "+OR+".join(f"cat:{c}" for c in cs_cats)
#     # submittedDate range for the target day in YYYYMMDDhhmm format: from 00:00 to 23:59
#     date_str = date_obj.strftime("%Y%m%d")
#     date_start = f"{date_str}0000"
#     date_end = f"{date_str}2359"
#     date_query = f"submittedDate:[{date_start}+TO+{date_end}]"
#     full_query = f"{cat_query}+AND+{date_query}"
#     feed_url = (
#         f"https://export.arxiv.org/api/query?"
#         f"search_query={full_query}&sortBy=submittedDate&sortOrder=ascending&max_results=2000"
#     )
#     logging.debug(f"Using date-range arXiv query: {feed_url}")
#     try:
#         feed = feedparser.parse(feed_url)
#     except Exception as e:
#         logging.error(f"Error fetching or parsing arXiv feed: {e}")
#         return []
#     if feed.bozo:
#         logging.warning(f"Feed may be ill-formed: {feed.bozo_exception}")
#     entries = getattr(feed, 'entries', [])
#     papers: List[Dict[str, str]] = []
#     for entry in entries:
#         # published_parsed is a time.struct_time
#         try:
#             pub_tm = entry.published_parsed
#             pub_dt = dt.date(pub_tm.tm_year, pub_tm.tm_mon, pub_tm.tm_mday)
#         except Exception:
#             continue
#         if pub_dt != date_obj:
#             continue
#         arxiv_id = entry.id.rsplit('/', 1)[-1]
#         papers.append({
#             "title": entry.title,
#             "abstract": entry.summary.replace("\n", " "),
#             "id": arxiv_id,
#             "url": entry.link,
#             "published": entry.published,
#         })
#     logging.info(f"Fetched {len(papers)} papers from arXiv for {date} via date-range query.")
#     return papers 
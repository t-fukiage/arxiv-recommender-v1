from __future__ import annotations

import datetime as dt
import logging
import pathlib
import re # 正規表現ライブラリを追加
from typing import List, Dict, Any

import bibtexparser # type: ignore
# import feedparser # type: ignore
import requests # Add requests for bioRxiv API

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

                # Extract and format authors
                authors_list = []
                try:
                    # Check if persons attribute exists (bibtexparser v2 standard)
                    if hasattr(entry, 'persons') and 'author' in entry.persons:
                        for person in entry.persons['author']:
                            # Construct full name from parts
                            first = " ".join(person.first)
                            von = " ".join(person.von)
                            last = " ".join(person.last)
                            jr = " ".join(person.jr)
                            # Basic name formatting, adjust as needed
                            name_parts = [part for part in [first, von, last, jr] if part]
                            authors_list.append(" ".join(name_parts))
                        if authors_list:
                             logging.debug(f"Successfully parsed authors using 'persons' for entry {entry_id_for_log}")


                    # If persons parsing didn't work or 'persons' doesn't exist, try fallback
                    if not authors_list:
                        logging.debug(f"Attempting fallback author parsing for entry {entry_id_for_log}")
                        author_field = entry.fields_dict.get('author')
                        if author_field and author_field.value:
                            # Simple split by ' and ' as fallback
                            raw_authors = author_field.value
                            # Basic cleaning: remove potential excessive whitespace/newlines
                            raw_authors = re.sub(r'\s+', ' ', raw_authors).strip()
                            authors_list = [name.strip() for name in raw_authors.split(' and ')]
                            if authors_list:
                                logging.debug(f"Used fallback author parsing for entry {entry_id_for_log}, found: {authors_list}")
                            else:
                                 logging.warning(f"Fallback author parsing failed for entry {entry_id_for_log}, raw value: '{raw_authors}'")

                        else:
                             logging.debug(f"No author information found for entry {entry_id_for_log} (checked persons and fields_dict['author'])")


                except Exception as author_e:
                    # Catch other potential errors during name formatting etc.
                    logging.warning(f"Error processing authors for entry {entry_id_for_log}: {author_e}", exc_info=True)
                    authors_list = [] # Ensure it's an empty list on error

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
                    "authors": authors_list, # Add formatted authors
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

def fetch_biorxiv(cfg: dict, date_str: str | None = None) -> List[Dict[str, Any]]:
    """
    Fetches papers from bioRxiv based on categories and date settings in cfg.
    Uses the bioRxiv API: https://api.biorxiv.org/
    """
    biorxiv_cfg = cfg.get('fetch', {}).get('biorxiv', {})
    target_categories = biorxiv_cfg.get('categories', [])
    if not target_categories:
        logging.warning("No bioRxiv categories specified in config. Skipping bioRxiv fetch.")
        return []
    
    days_back = biorxiv_cfg.get('days_back', 1)
    max_results_per_category = biorxiv_cfg.get('max_results_per_category', 50)

    if date_str is None or date_str.lower() == 'today':
        # Calculate date based on days_back from today
        end_date_obj = dt.datetime.utcnow().date()
        start_date_obj = end_date_obj - dt.timedelta(days=days_back)
    else:
        try:
            # Use the provided date_str as the single day to query
            # The API expects start_date and end_date for a range.
            # For a single day, both will be the same.
            single_date_obj = dt.date.fromisoformat(date_str)
            start_date_obj = single_date_obj
            end_date_obj = single_date_obj
            logging.info(f"Targeting bioRxiv papers for specified date (UTC): {single_date_obj.isoformat()}")
        except ValueError:
            logging.error(f"Invalid date format for bioRxiv: '{date_str}'. Please use YYYY-MM-DD or 'today'. Defaulting to {days_back} day(s) back from today.")
            end_date_obj = dt.datetime.utcnow().date()
            start_date_obj = end_date_obj - dt.timedelta(days=days_back)

    start_date_api_str = start_date_obj.isoformat()
    end_date_api_str = end_date_obj.isoformat()
    
    logging.info(f"Querying bioRxiv for categories: {target_categories} from {start_date_api_str} to {end_date_api_str}")

    all_papers = []
    seen_dois = set() 

    for category in target_categories:
        encoded_category = urllib.parse.quote(category)
        logging.info(f"Fetching bioRxiv papers for category: '{category}' (encoded: '{encoded_category}')")
        
        cursor = 0
        retrieved_count_for_category = 0
        
        # Loop for pagination for the current category
        while retrieved_count_for_category < max_results_per_category:
            # Endpoint: https://api.biorxiv.org/details/[server]/[startDate]/[endDate]/[cursor]/[format]?category=[cat]
            api_url = f"https://api.biorxiv.org/details/biorxiv/{start_date_api_str}/{end_date_api_str}/{cursor}/json?category={encoded_category}"
            logging.debug(f"Querying bioRxiv API: {api_url}")

            try:
                response = requests.get(api_url, timeout=30)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching data from bioRxiv API ({api_url}): {e}")
                break # Stop fetching for this category on error
            except ValueError as e: # JSON decoding errors
                logging.error(f"Error decoding JSON from bioRxiv API ({api_url}): {e}")
                break # Stop fetching for this category on error

            if not data or 'collection' not in data or not data['collection']:
                logging.info(f"No more papers found from bioRxiv for category '{category}' with current date range/cursor.")
                break # No more papers for this category with this cursor

            current_batch = data['collection']

            for entry in current_batch:
                if retrieved_count_for_category >= max_results_per_category:
                    break # Reached the limit for this category

                # Category check (though API should have filtered, good for sanity check or if API behavior varies)
                # paper_api_category = entry.get('category', '')
                # if paper_api_category.lower() != category.lower():
                #     logging.warning(f"API returned paper with category '{paper_api_category}' when querying for '{category}'. Skipping DOI {entry.get('doi')}")
                #     continue
            
                if not all(k in entry for k in ('doi', 'title', 'abstract', 'date', 'authors')):
                    logging.warning(f"Skipping bioRxiv entry due to missing essential fields: {entry.get('doi', 'N/A')}")
                    continue
                
                doi = entry['doi']
                if doi in seen_dois:
                    logging.debug(f"Skipping duplicate bioRxiv paper (DOI {doi} already seen, possibly from another category fetch).")
                    continue
                seen_dois.add(doi)

                try:
                    authors_list = []
                    if isinstance(entry.get('authors'), list):
                        for author_entry in entry['authors']:
                            if isinstance(author_entry, dict) and 'name' in author_entry:
                                authors_list.append(author_entry['name'])
                            elif isinstance(author_entry, str):
                                authors_list.append(author_entry)
                            else:
                                authors_list.append(str(author_entry))
                    elif isinstance(entry.get('authors'), str):
                        authors_list = [name.strip() for name in entry['authors'].split(';')]

                    published_date_str = entry.get('date')
                    p_date = dt.datetime.strptime(published_date_str, "%Y-%m-%d").isoformat()

                    all_papers.append({
                        "id": doi,
                        "title": entry['title'],
                        "abstract": entry['abstract'].replace("\\n", " "),
                        "published": p_date,
                        "url": f"https://www.biorxiv.org/content/{doi}",
                        "authors": authors_list,
                        "primary_category": entry.get('category'),
                        "source": "bioRxiv",
                        "doi": doi,
                        "server": entry.get('server', 'biorxiv'),
                        "version": entry.get('version')
                    })
                    retrieved_count_for_category += 1
                except Exception as e:
                    logging.warning(f"Error processing bioRxiv entry {doi}: {e}", exc_info=True)
                    continue
            
            if retrieved_count_for_category >= max_results_per_category:
                logging.info(f"Reached max_results_per_category ({max_results_per_category}) for category '{category}'.")
                break # Break from pagination loop for this category

            # Update cursor for next page for this category
            messages = data.get('messages', [{}])
            if messages and messages[0].get('cursor') is not None:
                new_cursor = messages[0]['cursor']
                if isinstance(new_cursor, str):
                    new_cursor = int(new_cursor)
                
                if new_cursor == 0 or new_cursor == cursor: 
                    logging.debug(f"BioRxiv API cursor for category '{category}' indicates no more results or is not advancing.")
                    break # No more pages for this category
                cursor = new_cursor
            else:
                logging.debug(f"No cursor information from bioRxiv for category '{category}', assuming no more pages.")
                break # No more pages for this category
    
    logging.info(f"Fetched {len(all_papers)} total papers from bioRxiv across specified categories.")
    return all_papers

def fetch_arxiv(cfg: dict, date_str: str | None = None):
    """Fetch papers from arxiv.org."""
    arxiv_cfg = cfg.get('fetch', {}).get('arxiv', {})
    target_categories = arxiv_cfg.get('categories', CS_CATS) # Default to CS_CATS if not in config
    days_back = arxiv_cfg.get('days_back', 1)
    # arXiv API's max_results is for the entire query, not per category.
    # The config name 'max_results_per_category' for arxiv is a bit misleading here.
    # We'll use it as the global max_results for the arXiv query.
    max_total_results = arxiv_cfg.get('max_results_per_category', 2000)


    if not target_categories:
        logging.warning("No arXiv categories specified in config. Defaulting to CS_CATS.")
        target_categories = CS_CATS
        
    if date_str is None or date_str.lower() == 'today':
        # Query for papers submitted on the day before (UTC)
        # arXiv API query for specific date range is complex; using submittedDate sort
        # and fetching recent papers then filtering by published date is more common.
        # Here, we adjust our query target date.
        query_date_obj = (dt.datetime.utcnow() - dt.timedelta(days=days_back)).date()
        logging.info(f"Targeting arXiv papers submitted around {days_back} day(s) ago (UTC): {query_date_obj.isoformat()}")
    else:
        try:
            query_date_obj = dt.date.fromisoformat(date_str)
            logging.info(f"Targeting arXiv papers submitted around specified date (UTC): {query_date_obj.isoformat()}")
        except ValueError:
            logging.error(f"Invalid date format for arXiv: '{date_str}'. Please use YYYY-MM-DD or 'today'. Defaulting to {days_back} day(s) ago.")
            query_date_obj = (dt.datetime.utcnow() - dt.timedelta(days=days_back)).date()

    # Construct query for arXiv API
    # The arXiv API date filtering is on submission date.
    # We are looking for papers *submitted* on query_date_obj.
    # A common strategy is to search broadly for recent papers in categories and then filter locally.
    # However, trying to be more specific with the query first.
    # query = " OR ".join([f"cat:{cat}" for cat in target_categories])
    # The Search object handles category combination internally if multiple are given in the query string like "cat:cs.AI OR cat:cs.LG"
    # Or it can take a list of categories in some client versions/methods.
    # The `arxiv` library's `Search` object expects a single query string.
    
    # We want papers whose *submission date* is within our target window.
    # arXiv API returns 'published' and 'updated' dates. 'published' is the first version.
    # We will fetch recent papers and then filter by the 'published' date being query_date_obj.
    
    query_parts = []
    for cat in target_categories:
        query_parts.append(f"cat:{cat}")
    full_query = " OR ".join(query_parts)
    
    logging.info(f"Constructed arXiv query: {full_query} (will sort by submitted date and filter by published date locally)")

    search = arxiv.Search(
        query=full_query,
        max_results=max_total_results, # This is total max results for the query
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    
    client = arxiv.Client(
        page_size = min(max_total_results, 1000), # Max page size for client
        delay_seconds = 3,
        num_retries = 3
    )

    papers = []
    fetched_count = 0
    # Iterate through results from the search generator
    for r in client.results(search):
        # The result's 'published' field is a datetime object.
        # We want to match against our target query_date_obj (a date object).
        if r.published.date() == query_date_obj:
            papers.append({
                "id": r.entry_id.rsplit("/", 1)[-1],
                "title": r.title,
                "abstract": r.summary.replace("\\n", " "),
                "published": r.published.isoformat(), # Store as ISO string
                "url": r.entry_id, 
                "authors": [str(author) for author in r.authors],
                "primary_category": r.primary_category,
                "comments": r.comment if hasattr(r, 'comment') and r.comment else None,
                "source": "arXiv"
            })
            fetched_count +=1
        elif r.published.date() < query_date_obj:
            # Since results are sorted by newest submission, if we pass our target published date,
            # we can stop early.
            logging.debug(f"Stopping arXiv fetch as paper published date {r.published.date()} is before target {query_date_obj}")
            break
        if fetched_count >= max_total_results : # Safety break if local filter still yields too many
             logging.info(f"Reached max_total_results ({max_total_results}) for arXiv after local date filtering.")
             break


    logging.info(f"Fetched {len(papers)} papers from arXiv for categories {target_categories} published on {query_date_obj.isoformat()}")
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
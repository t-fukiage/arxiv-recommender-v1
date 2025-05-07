import os
import time
import logging
import google.generativeai as genai
import arxiv
from pathlib import Path
import hashlib
import requests
import tempfile # For handling temporary PDF files
# from pdfminer.high_level import extract_text

logger = logging.getLogger(__name__)

# Default prompt templates for chat and PDF modes
DEFAULT_PROMPT_TEMPLATE = None
DEFAULT_FILES_API_PROMPT_TEMPLATE = None

# Simple cache directories check and creation
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

# Fetch and extract full-text PDF
# Creates a 'pdfs' subdirectory under the cache_dir
# Caches both the PDF and the extracted text
def fetch_and_extract_fulltext(arxiv_id: str, pdf_url: str, cache_dir: Path) -> str | None:
    pdf_cache_dir = cache_dir / 'pdfs'
    ensure_dir(pdf_cache_dir)
    safe_id = arxiv_id.replace('/', '_')
    pdf_path = pdf_cache_dir / f"{safe_id}.pdf"
    txt_path = pdf_cache_dir / f"{safe_id}.txt"

    # Return cached text if available
    if txt_path.exists():
        try:
            return txt_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Error reading cached full-text for {arxiv_id}: {e}")

    # Download PDF if not cached
    try:
        if not pdf_path.exists():
            base_session_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
            }
            
            session = requests.Session()
            session.headers.update(base_session_headers)
            resp = None # Initialize resp

            if "biorxiv.org/content" in pdf_url:
                html_url = pdf_url.replace(".full.pdf", ".full-text")
                
                html_headers = {
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                    "Referer": "https://www.biorxiv.org/" 
                }
                try:
                    logger.debug(f"Preloading HTML for {arxiv_id}: {html_url}")
                    session.get(html_url, headers=html_headers, timeout=15)
                except Exception as e:
                    logger.warning(f"Error preloading HTML page {html_url} for {arxiv_id}: {e}")

                pdf_headers = {
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8,application/pdf;q=0.9",
                    "Referer": html_url 
                }
                logger.debug(f"Fetching PDF for {arxiv_id}: {pdf_url}")
                resp = session.get(pdf_url, headers=pdf_headers, timeout=30)
            
            else: 
                non_biorxiv_headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    "Accept": "application/pdf,*/*;q=0.8",
                }
                logger.debug(f"Fetching non-bioRxiv PDF for {arxiv_id}: {pdf_url}")
                resp = requests.get(pdf_url, headers=non_biorxiv_headers, timeout=30)
            
            resp.raise_for_status()
            pdf_path.write_bytes(resp.content)
    except Exception as e:
        logger.error(f"Error downloading PDF for {arxiv_id} from {pdf_url}: {e}")
        return None

    # Extract text from PDF
    try:
        text = extract_text(str(pdf_path))
        txt_path.write_text(text, encoding='utf-8')
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF for {arxiv_id}: {e}")
        return None

# Helper to download PDF bytes for direct PDF-based generation
def fetch_pdf_bytes(pdf_url: str) -> bytes | None:
    try:
        base_session_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
        }
        
        session = requests.Session()
        session.headers.update(base_session_headers)
        resp = None # Initialize resp

        if "biorxiv.org/content" in pdf_url:
            html_url = pdf_url.replace(".full.pdf", ".full-text")
            
            html_headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Referer": "https://www.biorxiv.org/" 
            }
            try:
                logger.debug(f"Preloading HTML for PDF URL: {html_url}")
                session.get(html_url, headers=html_headers, timeout=15)
            except Exception as e:
                logger.warning(f"Error preloading HTML page {html_url} for PDF URL {pdf_url}: {e}")

            pdf_headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8,application/pdf;q=0.9",
                "Referer": html_url 
            }
            logger.debug(f"Fetching PDF: {pdf_url}")
            resp = session.get(pdf_url, headers=pdf_headers, timeout=30)
        
        else: 
            non_biorxiv_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "application/pdf,*/*;q=0.8",
            }
            logger.debug(f"Fetching non-bioRxiv PDF: {pdf_url}")
            resp = requests.get(pdf_url, headers=non_biorxiv_headers, timeout=30)
        
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.error(f"Error downloading PDF bytes from {pdf_url}: {e}")
        return None

# Function to generate explanation using Gemini API
def generate_explanation_gemini(
    paper_title: str,
    paper_abstract: str,
    paper_pdf_bytes: bytes | None,
    config: dict,
    api_key: str | None,
) -> str | None:
    """Generates an explanation for a paper using the Gemini API."""
    # Declare globals used within the function scope
    global DEFAULT_PROMPT_TEMPLATE, DEFAULT_FILES_API_PROMPT_TEMPLATE
    if not api_key:
        logger.error("GAI_API_KEY not found in environment variables.")
        return None

    genai.configure(api_key=api_key)

    target_language = config.get('target_language', 'English')
    # Files API generation: Upload PDF and call with File object
    if config.get('use_files_api', False) and paper_pdf_bytes:
        files_api_model = config.get('files_api_model', 'gemini-1.5-pro-latest')
        logger.info(f"Attempting explanation generation using Files API with model: {files_api_model}")
        temp_pdf_file = None
        uploaded_file = None
        try:
            # Create a temporary file to upload
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf_file:
                temp_pdf_file.write(paper_pdf_bytes)
                temp_pdf_path = temp_pdf_file.name
            logger.debug(f"Uploading temporary PDF: {temp_pdf_path}")
            uploaded_file = genai.upload_file(path=temp_pdf_path)
            logger.debug(f"Uploaded file URI: {uploaded_file.uri}")

            # Prepare prompt and generation config for Files API
            prompt_tmpl = config.get('files_api_prompt_template', DEFAULT_FILES_API_PROMPT_TEMPLATE)
            if prompt_tmpl is None: # Define default if needed
                DEFAULT_FILES_API_PROMPT_TEMPLATE = "この PDF 論文を{target_language}で 300 字程度で要約してください。"
                prompt_tmpl = DEFAULT_FILES_API_PROMPT_TEMPLATE
            user_prompt = prompt_tmpl.format(target_language=target_language)
            gen_cfg = config.get('files_api_generation_config', {})
            generation_config = genai.types.GenerationConfig(
                temperature=gen_cfg.get('temperature', 0.25),
                top_p=gen_cfg.get('top_p', 0.9),
                max_output_tokens=gen_cfg.get('max_output_tokens', 1024)
            )
            # Call generate_content with prompt and uploaded file
            logger.debug("Calling generate_content with Files API prompt and file URI...")
            model = genai.GenerativeModel(model_name=files_api_model)
            response = model.generate_content(
                contents=[user_prompt, uploaded_file], # Order per snippet
                generation_config=generation_config
            )
            # Return explanation text
            explanation = getattr(response, 'text', '').strip()
            logger.info(f"Successfully generated explanation via Files API (length: {len(explanation)})")
            if uploaded_file:
                try:
                    logger.debug(f"Deleting uploaded file: {uploaded_file.name}")
                    genai.delete_file(name=uploaded_file.name)
                    logger.info(f"Successfully deleted uploaded file: {uploaded_file.name}")
                except Exception as e:
                    logger.error(f"Error deleting uploaded file {uploaded_file.name}: {e}")
            return explanation
        except Exception as e:
            logger.error(f"Error during Files API explanation generation: {e}")
            return None # Fallback to text-based method below
        finally:
            # Clean up temporary file
            if temp_pdf_file and os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                    logger.debug(f"Removed temporary PDF: {temp_pdf_path}")
                except OSError as e:
                    logger.error(f"Error removing temporary PDF {temp_pdf_path}: {e}")

    # --- Fallback to Text-based Generation (using Abstract) --- 
    logger.info("Falling back to abstract-based text explanation...")
    model_name = config.get('model_name', 'gemini-1.5-flash-latest') # Fallback model
    prompt_template = config.get('prompt_template', DEFAULT_PROMPT_TEMPLATE)
    # Define default text template if not set
    if DEFAULT_PROMPT_TEMPLATE is None:
        DEFAULT_PROMPT_TEMPLATE = f"""以下の論文について、主要な貢献と新規性を中心に{target_language}で簡潔に解説してください。
専門用語は必要に応じて簡単な言葉で補足してください。

Title: {{title}}

Abstract: {{abstract}}"""

    # Validate template
    needed = ['{title}', '{abstract}', '{target_language}']
    if not isinstance(prompt_template, str) or not all(tok in prompt_template for tok in needed):
        logger.warning(f"Invalid text prompt_template, using default. Template was: {prompt_template}")
        prompt_template = DEFAULT_PROMPT_TEMPLATE

    # Build prompt
    prompt = prompt_template.format(
        title=paper_title,
        abstract=paper_abstract,
        target_language=target_language
    )

    # Call Gemini API (Text-based)
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.2) # Simple config for fallback
        )
        if response.candidates and response.candidates[0].content.parts:
            explanation = response.candidates[0].content.parts[0].text.strip()
            logger.debug(f"Successfully generated fallback explanation (length: {len(explanation)})")
            return explanation
        else:
            logger.warning(f"Fallback Gemini response was empty or blocked for prompt: {prompt[:100]}...")
            return None
    except Exception as e:
        logger.error(f"Error generating fallback text explanation via Gemini: {e}")
        return None

# Main function to get explanation (handles caching)
def get_explanation(paper_id: str, source: str, config: dict) -> str | None:
    """
    Gets the explanation for a given paper ID and source (arXiv or bioRxiv).
    Checks cache first, otherwise fetches paper details, generates explanation, and caches it.
    """
    explanation_config = config.get('explanation', {})
    if not explanation_config.get('enable', False):
        logger.info("Explanation feature is disabled in config.")
        return None

    main_cache_dir = Path(config.get('cache_dir', 'cache'))
    explanation_cache_subdir = explanation_config.get('cache_dir', 'explanation_cache')
    cache_dir = main_cache_dir / explanation_cache_subdir
    ensure_dir(cache_dir)

    # Create a unique cache key based on ID, source, model, and language
    model_name = explanation_config.get('model_name', 'gemini-1.5-flash-latest')
    target_language = explanation_config.get('target_language', 'English')
    config_hash = hashlib.md5(f"{model_name}-{target_language}".encode()).hexdigest()[:8]
    # Include source in cache filename to differentiate between arXiv ID and DOI if they could be identical
    cache_filename = f"{source}_{paper_id.replace('/', '_')}_{config_hash}.txt"
    cache_file = cache_dir / cache_filename

    # 1. Check cache
    if cache_file.exists():
        logger.debug(f"Cache hit for explanation: {cache_file}")
        try:
            return cache_file.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Error reading cache file {cache_file}: {e}")
            # Proceed to generate if cache is corrupted

    logger.info(f"Cache miss for explanation: {cache_file}. Generating...")

    # 2. Fetch paper details (Title and Abstract)
    title = None
    abstract = None
    pdf_url = None # Initialize pdf_url
    paper_version = None # For bioRxiv PDF URL construction

    if source.lower() == 'arxiv':
        try:
            search = arxiv.Search(id_list=[paper_id], max_results=1)
            results = list(search.results())
            if not results:
                logger.error(f"Could not find paper with arXiv ID: {paper_id}")
                return None
            paper_obj = results[0]
            title = paper_obj.title
            abstract = paper_obj.summary.replace('\n', ' ')
            pdf_url = getattr(paper_obj, 'pdf_url', None)
            logger.debug(f"Fetched details for arXiv:{paper_id}: '{title[:50]}...'")
        except Exception as e:
            logger.error(f"Error fetching paper details for arXiv ID {paper_id} from arXiv API: {e}")
            return None
    elif source.lower() == 'biorxiv':
        # Fetch details from bioRxiv API for DOI
        # This assumes paper_id is a DOI for bioRxiv
        try:
            # Use the bioRxiv API to get details for a single DOI
            # https://api.biorxiv.org/details/[server]/[DOI]/na/[format]
            biorxiv_api_url = f"https://api.biorxiv.org/details/biorxiv/{paper_id}/na/json"
            response = requests.get(biorxiv_api_url, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data and 'collection' in data and data['collection']:
                entry = data['collection'][0]
                title = entry.get('title')
                abstract = entry.get('abstract', '').replace('\n', ' ')
                paper_version = entry.get('version')
                # Construct bioRxiv PDF URL: https://www.biorxiv.org/content/{DOI}v{version}.full.pdf
                if paper_version:
                    pdf_url = f"https://www.biorxiv.org/content/{paper_id}v{paper_version}.full.pdf"
                else:
                    # Fallback if version is not available, try without version (might get latest or fail)
                    pdf_url = f"https://www.biorxiv.org/content/{paper_id}.full.pdf"
                logger.debug(f"Fetched details for bioRxiv:{paper_id} (v{paper_version}): '{title[:50]}...'")
            else:
                logger.error(f"Could not find paper details for DOI {paper_id} on bioRxiv or API response malformed.")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching paper details for DOI {paper_id} from bioRxiv API: {e}")
            return None
        except ValueError as e: # JSON decoding error
            logger.error(f"Error decoding JSON from bioRxiv API for DOI {paper_id}: {e}")
            return None
    else:
        logger.error(f"Unsupported source: {source} for paper ID: {paper_id}")
        return None

    if not title or not abstract:
        logger.error(f"Failed to get title or abstract for {source}:{paper_id}. Cannot generate explanation.")
        return None

    # 3. Fetch PDF bytes if Files API is enabled
    pdf_bytes = None
    if explanation_config.get('use_files_api', False):
        if pdf_url:
            pdf_bytes = fetch_pdf_bytes(pdf_url)
            if not pdf_bytes:
                logger.warning(f"PDF download failed from {pdf_url}; cannot use Files API. Will fallback to text-based.")
        else:
            logger.warning(f"PDF URL not available for {source}:{paper_id}; cannot use Files API.")

    # 4. Generate explanation
    api_key = os.getenv("GAI_API_KEY")
    explanation = generate_explanation_gemini(
        paper_title=title,
        paper_abstract=abstract,
        paper_pdf_bytes=pdf_bytes,
        config=explanation_config,
        api_key=api_key
    )

    # 4. Save to cache if successful
    if explanation:
        logger.info(f"Saving generated explanation to cache: {cache_file}")
        try:
            cache_file.write_text(explanation, encoding='utf-8')
        except Exception as e:
            logger.error(f"Error writing cache file {cache_file}: {e}")
    else:
        logger.warning(f"Failed to generate explanation for {source}:{paper_id}.")

    return explanation

# Example usage (for testing)
if __name__ == '__main__':
    # Setup basic logging for testing
    logging.basicConfig(level=logging.DEBUG)

    # Mock config for testing
    mock_config = {
        'cache_dir': 'cache',
        'explanation': {
            'enable': True,
            'model_name': 'gemini-1.5-flash-latest', # Or 'gemini-1.5-pro-latest' for Files API testing
            'target_language': 'Japanese',
            'cache_dir': 'explanation_cache',
            'prompt_template': """以下の論文について、主要な貢献と新規性を中心に{target_language}で3-4文で簡潔に解説してください。

Title: {title}

Abstract: {abstract}""",
            # To test Files API mode, uncomment the following and ensure a capable model is set
            # 'use_files_api': True,
            # 'files_api_model': 'gemini-1.5-pro-latest',
            # 'files_api_prompt_template': "この PDF 論文を{target_language}で 300 字程度で要約してください。"
        }
    }

    # Ensure API key is set as environment variable: export GAI_API_KEY='YOUR_KEY'
    
    # --- Test arXiv ---
    test_arxiv_id = '2305.15334' # Example arXiv paper ID
    print(f"Attempting to get explanation for arXiv ID: {test_arxiv_id}")
    explanation_text_arxiv = get_explanation(test_arxiv_id, 'arxiv', mock_config)

    if explanation_text_arxiv:
        print("\n--- Explanation (arXiv) ---")
        print(explanation_text_arxiv)
        print("\n---------------------------")
    else:
        print(f"\nFailed to get explanation for arXiv ID: {test_arxiv_id}.")

    # Test arXiv cache retrieval
    print(f"\nAttempting to get explanation for arXiv ID {test_arxiv_id} again (should be cached)...")
    explanation_text_arxiv_cached = get_explanation(test_arxiv_id, 'arxiv', mock_config)
    if explanation_text_arxiv_cached:
        print("\n--- Cached Explanation (arXiv) ---")
        print(explanation_text_arxiv_cached)
        print("\n--------------------------------")
    else:
         print(f"\nFailed to get cached explanation for arXiv ID: {test_arxiv_id}.")

    print("\n=====================================================\n")

    # --- Test bioRxiv ---
    # Replace with a valid bioRxiv DOI that has a PDF available for Files API testing if enabled
    test_biorxiv_doi = "10.1101/2023.10.13.562298" 
    print(f"Attempting to get explanation for bioRxiv DOI: {test_biorxiv_doi}")
    explanation_text_biorxiv = get_explanation(test_biorxiv_doi, 'biorxiv', mock_config)

    if explanation_text_biorxiv:
        print("\n--- Explanation (bioRxiv) ---")
        print(explanation_text_biorxiv)
        print("\n-----------------------------")
    else:
        print(f"\nFailed to get explanation for bioRxiv DOI: {test_biorxiv_doi}.")

    # Test bioRxiv cache retrieval
    print(f"\nAttempting to get explanation for bioRxiv DOI {test_biorxiv_doi} again (should be cached)...")
    explanation_text_biorxiv_cached = get_explanation(test_biorxiv_doi, 'biorxiv', mock_config)
    if explanation_text_biorxiv_cached:
        print("\n--- Cached Explanation (bioRxiv) ---")
        print(explanation_text_biorxiv_cached)
        print("\n----------------------------------")
    else:
        print(f"\nFailed to get cached explanation for bioRxiv DOI: {test_biorxiv_doi}.") 
import sys
import os
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# Adjust sys.path to import from core
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parents[2] # Assumes server/proxy.py is two levels down from root
sys.path.insert(0, str(project_root / 'src'))

# Now import core modules
try:
    from arxiv_recommender.core.utils import load_config, setup_logger
    from arxiv_recommender.core.explain import get_explanation
except ImportError as e:
    print(f"Error importing core modules: {e}", file=sys.stderr)
    print("Ensure the script is run from the project root or PYTHONPATH is set correctly.", file=sys.stderr)
    sys.exit(1)

app = Flask(__name__)
# Enable CORS for requests from file:// or localhost where static HTML might be served
CORS(app) 

# Load configuration globally? Or per request?
# Loading per request is safer if config can change, but less efficient.
# Loading globally assumes config doesn't change while proxy is running.
config_path = project_root / 'config.yaml' 
cfg = {}

@app.before_request
def load_app_config():
    global cfg
    try:
        # Reload config on each request to pick up changes (can be optimized)
        cfg = load_config(config_path)
        # Ensure logger is set up based on config/env
        setup_logger(cfg.get('log_level', os.getenv('LOGLEVEL', 'INFO')))
        # Ensure API key is available (important!)
        if not os.getenv("GAI_API_KEY"):
             # Check the old name too, just in case
             if not os.getenv("GEMINI_API_KEY"):
                   logging.warning("GAI_API_KEY (or GEMINI_API_KEY) environment variable not set. Explanation generation will likely fail.")
             else:
                   # If old key exists, maybe set the new one for consistency within get_explanation?
                   os.environ["GAI_API_KEY"] = os.getenv("GEMINI_API_KEY")

    except Exception as e:
        logging.error(f"Failed to load config {config_path}: {e}", exc_info=True)
        # Potentially return an error response here? For now, log it.


@app.route('/explain', methods=['GET'])
def explain_paper():
    arxiv_id = request.args.get('id')
    if not arxiv_id:
        return jsonify({'error': 'Missing required parameter: id'}), 400

    if not cfg:
         return jsonify({'error': 'Server configuration could not be loaded.'}), 500

    logging.info(f"Received request to explain arXiv ID: {arxiv_id}")

    try:
        # Use the imported get_explanation function
        explanation_text = get_explanation(arxiv_id, cfg)

        if explanation_text:
            logging.info(f"Successfully generated explanation for {arxiv_id}")
            return jsonify({'explanation': explanation_text})
        else:
            logging.warning(f"Failed to generate explanation for {arxiv_id} (returned None)")
            return jsonify({'error': f'Failed to generate explanation for {arxiv_id}. Check proxy server logs.'}), 500

    except Exception as e:
        logging.error(f"Exception during explanation generation for {arxiv_id}: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500

if __name__ == '__main__':
    print("Starting arXiv Recommender explanation proxy server...")
    print(f"Configuration loaded from: {config_path.resolve()}")
    print("API Key Check: Ensure GAI_API_KEY or GEMINI_API_KEY environment variable is set.")
    print("Listening on http://localhost:5001")
    print("Press CTRL+C to stop")
    # Run Flask dev server (suitable for local use)
    # Consider using a production server like Gunicorn/Waitress for more robust deployment
    app.run(host='localhost', port=5001, debug=False) # Turn debug=False for security 
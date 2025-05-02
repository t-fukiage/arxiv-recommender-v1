#!/bin/bash

# How to run this script:
# ./run_all_en.sh

# If you see an error like "zsh: permission denied: ./run_all.sh", add execute permission with:
#   chmod +x run_all.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration (Edit as needed) ---
# WARNING: Writing the API key directly into the script is not recommended for security reasons.
# If possible, remove this line and set the environment variable before running the script.
# Example: export GAI_API_KEY='...'
export GAI_API_KEY='YOUR_API_KEY_HERE' # <<< Replace with your actual API key!

# Path to your BibTeX file (Replace with the actual path)
BIB_FILE="my.bib" # <<< Please modify

# HTTP server port number (Change if 8001 is already in use)
HTTP_PORT=8001
# Proxy server port number (Matches proxy.py default)
PROXY_PORT=5001
# Output directory
OUTPUT_DIR="output"
# Log files
PROXY_LOG="proxy.log"
HTTP_LOG="http_server.log"

# --- Execution ---

# Remove existing log files (optional)
rm -f "$PROXY_LOG" "$HTTP_LOG"

echo "Step 1: Activating Conda environment (arxiv-reco-v1)..."
# Assumes `conda init bash` (or `conda init zsh`) has been run previously
source $(conda info --base)/etc/profile.d/conda.sh # Make conda command available in shell scripts
conda activate arxiv-reco-v1
echo "Conda environment OK."

echo "Step 2: Checking API key..."
if [ -z "$GAI_API_KEY" ] || [ "$GAI_API_KEY" == "YOUR_API_KEY_HERE" ]; then
  echo "Error: GAI_API_KEY is not set or is still the placeholder."
  echo "Replace 'YOUR_API_KEY_HERE' in the script with your actual key, or"
  echo "run 'export GAI_API_KEY=...' before executing the script."
  exit 1
fi
echo "API key OK."

echo "Step 3: Generating today's arXiv recommendations and HTML (cluster mode)..."
# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
# Run the recommendation script (using config.yaml)
python src/arxiv_recommender/core/main.py --bib "$BIB_FILE" --date today --cluster --config config.yaml
echo "HTML generation complete."

echo "Step 4: Starting proxy server in the background (localhost:$PROXY_PORT)..."
# Start proxy server in the background and redirect logs
python src/arxiv_recommender/server/proxy.py --port "$PROXY_PORT" > "$PROXY_LOG" 2>&1 &
proxy_pid=$!
echo "Proxy server started (PID: $proxy_pid). Log: $PROXY_LOG"

# Wait a bit for the server to start (optional)
sleep 2

echo "Step 5: Starting HTTP server in the background (localhost:$HTTP_PORT)..."
# Start HTTP server in the background and redirect logs
cd "$OUTPUT_DIR" # Change to output directory
python -m http.server "$HTTP_PORT" > "../$HTTP_LOG" 2>&1 & # Save log to parent directory
http_server_pid=$!
cd .. # Return to the original directory
echo "HTTP server started (PID: $http_server_pid). Log: $HTTP_LOG"

echo ""
echo "--- Ready --- "
echo "Open the following URL in your browser:"
echo "http://localhost:$HTTP_PORT"
echo ""

# Open browser automatically on macOS (Linux: xdg-open http://localhost:$HTTP_PORT)
# open "http://localhost:$HTTP_PORT"

# Wait for the user to press Enter
read -p "Press Enter in this terminal to stop the servers once you are done viewing the recommendations..."

echo ""
echo "Stopping servers..."

# Prevent script from stopping if kill fails
kill $proxy_pid 2>/dev/null || echo "Failed to stop proxy server ($proxy_pid) (maybe it was already stopped)."
kill $http_server_pid 2>/dev/null || echo "Failed to stop HTTP server ($http_server_pid) (maybe it was already stopped)."

echo "Servers stopped."
echo "" 
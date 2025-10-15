Ecommerce Product Recommendation System
======================================

This repository contains a prototype for an ecommerce product recommendation system, combining content-based and collaborative filtering approaches with a lightweight Flask-based UI and API. It also includes optional LLM-backed explanation features for enhanced recommendation interpretability.

Features
--------
- Hybrid Recommendation Engine: Combines content-based (using TF-IDF on product metadata) and collaborative filtering techniques for personalized product recommendations.
- Flask Web UI & API: A simple web interface at /main and RESTful API endpoints under /api/v1/ for recommendations and product search.
- Fuzzy Product Matching: Supports case-insensitive fuzzy matching for product names, with fallback to the dataset if not found in the database.
- LLM Explanations (Optional): Generates human-readable explanations for recommendations using providers like Gemini, OpenAI, or Ollama (if configured).
- SQLite Persistence: Stores product metadata and LLM explanation cache in a lightweight recommender.db.
- Sample Datasets: Includes clean_data.csv for demo/training and trending_products.csv for previews.

Project Structure
-----------------
- recommendation_code.py: Core recommendation utilities (data loading, content-based/collaborative/hybrid recommenders, persistence, LLM explainer).
- app.py: Flask application for the web UI and API endpoints.
- clean_data.csv: Primary dataset for demos and training.
- trending_products.csv: Optional dataset for trending product previews.
- recommender.db: SQLite database for product metadata and LLM cache (generated on first run, ignored by .gitignore).
- .gitignore: Ignores virtual environments, databases, logs, and optionally datasets.

Prerequisites
-------------
- Operating System: Windows (tested in PowerShell; other OS may work with minor adjustments)
- Python: 3.10 or higher
- Git: Optional, for cloning the repository

Set up (recommended)
Open PowerShell in the repository root and run:

```powershell
# create and activate virtual environment (one-time)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install -r .\requirements.txt
```

Run the CLI (quick validation)

```powershell
# activate env (if not already)
.\.venv\Scripts\Activate.ps1
python .\recommendation_code.py
```

Setup Instructions
------------------
1. Clone the Repository (if using Git):
   git clone https://github.com/jagjeet-singh04/Ecommerce-Product-recommendation-system.git
   cd Ecommerce-Product-recommendation-system

2. Create and Activate a Virtual Environment:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   ### or source .venv/bin/activate  ### Linux/Mac

3. Install Dependencies:
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt

Running the Project
-------------------

1. CLI Mode (Validation & Sample Recommendations)
Run the core recommendation script to validate the dataset, generate sample recommendations, and persist product metadata to recommender.db:

   .\.venv\Scripts\Activate.ps1  # Activate virtual environment
   python recommendation_code.py

This will output dataset validation details and example recommendations.

2. Flask Web UI & API
Start the Flask application to access the web UI and API:

   .\.venv\Scripts\Activate.ps1  # Activate virtual environment
   python app.py

- Web UI: Open http://127.0.0.1:5000/main in a browser to access the recommendation interface.
- API Endpoints: Use tools like curl, Postman, or PowerShell to interact with the API.

API Examples (PowerShell)
- Content-Based Recommendations:
  Invoke-RestMethod -Uri 'http://127.0.0.1:5000/api/v1/recommendations/content-based' -Method POST -Body (ConvertTo-Json @{product_name='OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath'; top_n=5}) -ContentType 'application/json'

- Collaborative Recommendations:
  Invoke-RestMethod -Uri 'http://127.0.0.1:5000/api/v1/recommendations/collaborative' -Method POST -Body (ConvertTo-Json @{user_id=1705736792; top_n=5}) -ContentType 'application/json'

- Hybrid Recommendations:
  Invoke-RestMethod -Uri 'http://127.0.0.1:5000/api/v1/recommendations/hybrid' -Method POST -Body (ConvertTo-Json @{user_id=1705736792; product_name='OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath'; top_n=5}) -ContentType 'application/json'

- Search Products:
  Invoke-RestMethod -Uri 'http://127.0.0.1:5000/api/v1/products/search?q=bubble' -Method GET

Sample Product Names for Testing
--------------------------------
Try these product names in the UI or API for reliable results:
- OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath
- OPI Nail Lacquer - Dont Bossa Nova Me Around - NL A65
- OPI Infinite Shine 2 Polish - ISL P33 - Alpaca My Bags
- La Vie Organic Cotton Top Sheet Panty Liners, Ultra Thin, 108 Count
- Egyptian Magic All Purpose Skin Cream 4 Ounce Jar All Natural Ingredients
- EcoTools Ultimate Shade Duo Eyeshadow Makeup Brush Set, 2pc
- Onyx Professional LED Mirror with In-Base Storage and Magnifying Mirror, White
- Schick Quattro for Women Razor Refill, Ultra Smooth, 8 Ct

Notes
-----
- Product Matching: The system uses case-insensitive fuzzy matching for product names. If a product is not found in recommender.db, it falls back to searching clean_data.csv.
- Tags for Content-Based Filtering: If the Tags column is missing in the dataset, the Flask app generates synthetic tags at runtime using Category, Brand, Name, and Description for TF-IDF-based recommendations.
- LLM Explanations: Optional feature requiring API keys for Gemini (GOOGLE_API_KEY or GEMINI_API_KEY), OpenAI (OPENAI_API_KEY), or Ollama (OLLAMA_URL/OLLAMA_KEY). Explanations are cached in recommender.db.

Troubleshooting
---------------
- NumPy/Pandas Binary Mismatch ("numpy.dtype size changed..."):
  Ensure the correct virtual environment is activated and update dependencies:
  .\.venv\Scripts\Activate.ps1
  pip install --upgrade pip setuptools wheel
  pip install -U numpy pandas
  ### Or pin versions: pip install 'numpy==1.26.*' 'pandas==2.1.*'

- "No recommendations available" in UI:
  Verify the product name matches or closely resembles one in the dataset (see sample product names above). The UI will suggest alternatives if no match is found.

- Flask Reloader Issues:
  Disable the Flask reloader for debugging complex state:
  python -m flask run --no-reload

Development Tips
----------------
- To exclude datasets from version control, uncomment the relevant lines in .gitignore.
- Contributions are welcome! Potential enhancements include:
  - Wiring UI suggestions into main.html.
  - Adding CLI flags for custom configurations.
  - Persisting synthesized tags to recommender.db on startup.

License
-------
This project is a prototype for demonstration and learning purposes. Add a license file if you plan to open-source it.

About
-----
Built as a prototype for an ecommerce product recommendation system, this project showcases hybrid recommendation techniques and a simple Flask-based interface.
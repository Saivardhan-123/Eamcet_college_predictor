import requests
from bs4 import BeautifulSoup
import pandas as pd
from flask import Flask, request, render_template, url_for, jsonify
import io
import re
import os
from datetime import datetime, timedelta
import hashlib
import json
from ml_rank_predictor import EamcetRankPredictor

app = Flask(__name__)

# The URL of the main EAPCET website.
EAPCET_MAIN_URL = "https://tgeapcet.nic.in/default.aspx"


# Cache settings
CACHE_DIR = os.path.join(os.getcwd(), 'cache')
CACHE_DURATION = 24 * 60 * 60  # 24 hours in seconds

# Ensure cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Initialize ML predictor
ml_predictor = EamcetRankPredictor()

# Model file paths (prefer specialized model first)
MODEL_PATHS = [
    'eamcet_rank_model_specialized.joblib',  # Specialized model with your actual data
    'eamcet_rank_model_improved_v2.joblib',  # Improved model v2
    'eamcet_rank_model_updated.joblib',      # Updated model with your data
    'eamcet_rank_model.joblib'              # Original model
]

# Load ML model if available
model_loaded = False
for model_path in MODEL_PATHS:
    if os.path.exists(model_path):
        try:
            ml_predictor.load_model(model_path)
            print(f"✅ ML model loaded successfully from {model_path}")
            model_loaded = True
            break
        except Exception as e:
            print(f"❌ Error loading {model_path}: {e}")
            continue

if not model_loaded:
    print(f"⚠️ No ML model found. Run updated_training_data.py to train an improved model.")

def get_current_academic_year():
    """
    Gets the current academic year based on the current date.
    Academic year runs from June to May of next year.
    """
    now = datetime.now()
    if now.month >= 6:  # June onwards is the new academic year
        return now.year
    else:  # January to May belongs to previous academic year
        return now.year - 1

def get_cache_file_path(url):
    """
    Generate a cache file path based on the URL.
    """
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f'data_{url_hash}.xlsx')

def is_cache_valid(cache_file_path):
    """
    Check if cache file exists and is within the cache duration.
    """
    if not os.path.exists(cache_file_path):
        return False
    
    file_age = datetime.now().timestamp() - os.path.getmtime(cache_file_path)
    return file_age < CACHE_DURATION

# A function to find the latest rank statement Special Phase link.
def find_special_phase_rank_statement_url(base_url):
    """
    Finds the URL for the "Last Rank Statement Special Phase" on the main EAPCET page.
    This function specifically looks for the Special Phase which has the most complete data.
    """
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Get current academic year
        current_year = get_current_academic_year()
        
        # First try to find the Special Phase link for the current year
        for year in [current_year, current_year - 1]:  # Try current year, then previous year
            # Look for "Special Phase" or "Final Phase" links
            special_phase_patterns = [
                f"TGEAPCET {year} Last Rank Statement Special Phase",
                f"TGEAPCET {year} Last Rank Statement Final Phase",
                f"Last Rank Statement Special Phase"
            ]
            
            for pattern in special_phase_patterns:
                links = soup.find_all('a', string=lambda s: s and pattern in s)
                for link in links:
                    href = link.get('href')
                    if href and href.endswith('.xlsx'):
                        print(f"Found Special Phase link for year {year}: {pattern}")
                        return requests.compat.urljoin(base_url, href), year
        
        # If no Special Phase found, try any Last Rank Statement for current year
        for year in [current_year, current_year - 1]:
            links = soup.find_all('a', string=lambda s: s and f"TGEAPCET {year}" in s and "Last Rank Statement" in s)
            for link in links:
                href = link.get('href')
                if href and href.endswith('.xlsx'):
                    print(f"Found general Last Rank Statement link for year {year}")
                    return requests.compat.urljoin(base_url, href), year
        
        print("Error: Could not find any Last Rank Statement Special Phase link. The website's layout might have changed.")
        return None, None
        
    except requests.exceptions.RequestException as e:
        print(f"Error accessing the main EAPCET website: {e}")
        return None, None

# A function to fetch and process the .xlsx data with caching.
def fetch_eapcet_data(url):
    """
    Fetches the .xlsx file from the given URL and converts its content
    into a pandas DataFrame. Uses caching to improve performance.
    """
    if not url:
        return pd.DataFrame(), None
    
    cache_file_path = get_cache_file_path(url)
    
    # Check if we have valid cached data
    if is_cache_valid(cache_file_path):
        try:
            print(f"Loading data from cache: {cache_file_path}")
            df = pd.read_excel(cache_file_path, header=0)
            if not df.empty:
                # Check if the first row contains actual headers, if not, use it as headers
                if df.iloc[0].astype(str).str.contains('Institute Name|Branch Name|OC|BC_A|SC|ST|EWS|Tuition').any():
                    # First row contains headers, use it
                    new_header = df.iloc[0]
                    df = df[1:]
                    df.columns = new_header
                
                # Clean up column names for consistent processing.
                df.columns = [str(col).replace(' ', '_').replace('\n', '_').upper().strip() for col in df.columns]
                
                # Remove any completely empty rows
                df = df.dropna(how='all')
                
                # Reset index
                df = df.reset_index(drop=True)
                
                return df, "cached"
        except Exception as e:
            print(f"Error loading cached data: {e}")
    
    # Fetch fresh data
    try:
        print(f"Fetching fresh data from: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save to cache first
        with open(cache_file_path, 'wb') as f:
            f.write(response.content)
        
        # Read the content of the Excel file directly into a DataFrame.
        # io.BytesIO is used to treat the downloaded data as an in-memory file.
        df = pd.read_excel(io.BytesIO(response.content), header=0)

        if not df.empty:
            # Check if the first row contains actual headers, if not, use it as headers
            if df.iloc[0].astype(str).str.contains('Institute Name|Branch Name|OC|BC_A|SC|ST|EWS|Tuition').any():
                # First row contains headers, use it
                new_header = df.iloc[0]
                df = df[1:]
                df.columns = new_header
            
            # Clean up column names for consistent processing.
            df.columns = [str(col).replace(' ', '_').replace('\n', '_').upper().strip() for col in df.columns]
            
            # Remove any completely empty rows
            df = df.dropna(how='all')
            
            # Reset index
            df = df.reset_index(drop=True)
            
            return df, "fresh"
        else:
            print("Error: The downloaded Excel file is empty.")
            return pd.DataFrame(), None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return pd.DataFrame(), None
    except Exception as e:
        print(f"An unexpected error occurred during processing the Excel file: {e}")
        return pd.DataFrame(), None

# Fallback function to use local CSV if online data fails
def get_fallback_data():
    """
    Fallback to local CSV file if online data fetching fails.
    """
    try:
        csv_path = os.path.join(os.getcwd(), 'data', 'eamcet_data.csv')
        if os.path.exists(csv_path):
            print("Using fallback local CSV data")
            df = pd.read_csv(csv_path)
            # Clean up column names for consistent processing.
            df.columns = [col.replace(' ', '_').upper() for col in df.columns]
            return df, "2024", "fallback"
    except Exception as e:
        print(f"Error loading fallback data: {e}")
    
    return pd.DataFrame(), None, None

# The main function to get the data
def get_data_for_prediction():
    try:
        url, year = find_special_phase_rank_statement_url(EAPCET_MAIN_URL)
        if url and year:
            df, data_source = fetch_eapcet_data(url)
            if not df.empty:
                return df, year, data_source
        
        # If online data fetch fails, try fallback
        print("Online data fetch failed, trying fallback...")
        return get_fallback_data()
        
    except Exception as e:
        print(f"Error in get_data_for_prediction: {e}")
        # Try fallback as last resort
        return get_fallback_data()

# This new helper function finds the correct column based on keywords
def find_column_by_keywords(df_columns, keywords):
    """
    Finds a column name in the DataFrame that contains all specified keywords.
    Keywords are case-insensitive.
    """
    for col in df_columns:
        if all(keyword in col.upper() for keyword in keywords):
            return col
    return None

@app.route("/", methods=["GET"])
def home():
    """Renders the home page."""
    return render_template("home.html")


# NEW: ML Rank Prediction Route
@app.route("/predict-rank", methods=["GET", "POST"])
def predict_rank():
    """
    Handles ML-based rank prediction from subject scores and integrated college prediction.
    """
    rank_prediction = None
    results = []
    has_results = False
    data_year = None
    data_source = None
    error_message = None
    step = 1  # Step 1: Enter scores, Step 2: Show results
    
    if request.method == "POST":
        action = request.form.get("action")
        
        if action == "predict_from_scores":
            # Step 1: Predict rank from subject scores
            try:
                math_score = float(request.form.get("math_score", 0))
                physics_score = float(request.form.get("physics_score", 0))
                chemistry_score = float(request.form.get("chemistry_score", 0))
                caste = request.form.get("caste")
                
                # Validate scores
                if not (0 <= math_score <= 80):
                    error_message = "Mathematics score must be between 0 and 80."
                elif not (0 <= physics_score <= 40):
                    error_message = "Physics score must be between 0 and 40."
                elif not (0 <= chemistry_score <= 40):
                    error_message = "Chemistry score must be between 0 and 40."
                elif not ml_predictor.is_trained:
                    error_message = "ML model not available. Please try again later."
                else:
                    # Predict rank using ML model
                    rank_prediction = ml_predictor.predict_rank(math_score, physics_score, chemistry_score)
                    step = 2  # Move to Step 2
                    
                    # Automatically predict colleges with the predicted rank
                    predicted_rank = rank_prediction['predicted_rank']
                    results, has_results, data_year, data_source = predict_colleges_for_rank(predicted_rank, caste)
                        
            except ValueError:
                error_message = "Please enter valid numeric scores."
            except Exception as e:
                error_message = f"Error during rank prediction: {str(e)}"
    
    return render_template("predict_rank.html", 
                         rank_prediction=rank_prediction,
                         results=results, 
                         has_results=has_results,
                         data_year=data_year,
                         data_source=data_source,
                         error_message=error_message,
                         step=step)

# NEW: API endpoint for ML rank prediction
@app.route("/api/predict-rank", methods=["POST"])
def api_predict_rank():
    """
    API endpoint to predict rank using subject scores.
    """
    try:
        data = request.get_json()
        math_score = float(data.get('math_score', 0))
        physics_score = float(data.get('physics_score', 0))
        chemistry_score = float(data.get('chemistry_score', 0))
        
        # Validate scores
        if not (0 <= math_score <= 80):
            return jsonify({
                "success": False,
                "error": "Mathematics score must be between 0 and 80."
            }), 400
            
        if not (0 <= physics_score <= 40):
            return jsonify({
                "success": False,
                "error": "Physics score must be between 0 and 40."
            }), 400
            
        if not (0 <= chemistry_score <= 40):
            return jsonify({
                "success": False,
                "error": "Chemistry score must be between 0 and 40."
            }), 400
            
        if not ml_predictor.is_trained:
            return jsonify({
                "success": False,
                "error": "ML model not available. Please try again later."
            }), 500
        
        # Predict rank using ML model
        prediction = ml_predictor.predict_rank(math_score, physics_score, chemistry_score)
        
        return jsonify({
            "success": True,
            **prediction
        })
        
    except ValueError:
        return jsonify({
            "success": False,
            "error": "Please provide valid numeric scores."
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500


def predict_colleges_for_rank(rank, caste):
    """
    Helper function to predict colleges for a given rank and caste.
    Returns: (results, has_results, data_year, data_source)
    """
    try:
        df, data_year, data_source = get_data_for_prediction()
        
        if df.empty:
            return [], False, data_year, data_source
        
        # Use the helper function to find the correct column names
        boys_column = find_column_by_keywords(df.columns, [caste.upper(), 'BOYS'])
        girls_column = find_column_by_keywords(df.columns, [caste.upper(), 'GIRLS'])
        
        # Check for other columns by keywords as well
        tuition_fee_col = find_column_by_keywords(df.columns, ['TUITION', 'FEE'])
        inst_name_col = find_column_by_keywords(df.columns, ['INSTITUTE', 'NAME'])
        branch_name_col = find_column_by_keywords(df.columns, ['BRANCH', 'NAME'])
        
        # This logic is more robust now and will only proceed if it finds all necessary columns
        if all([boys_column, girls_column, tuition_fee_col, inst_name_col, branch_name_col]):
            for col in [boys_column, girls_column]:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

            filtered_df = df[
                ((df[boys_column] >= rank) & (df[boys_column] != 0)) |
                ((df[girls_column] >= rank) & (df[girls_column] != 0))
            ].copy()

            if not filtered_df.empty:
                filtered_df['Rank_Diff'] = filtered_df.apply(
                    lambda row: min(abs(row[boys_column] - rank) if row[boys_column] != 0 else float('inf'),
                                    abs(row[girls_column] - rank) if row[girls_column] != 0 else float('inf')),
                    axis=1
                )
                filtered_df = filtered_df.sort_values(by='Rank_Diff', ascending=True)

                # Limit results to top 50 most relevant colleges
                results = filtered_df[[
                    inst_name_col, 
                    branch_name_col, 
                    tuition_fee_col, 
                    boys_column, 
                    girls_column
                ]].head(50).values.tolist()
                
                return results, True, data_year, data_source
        
        return [], False, data_year, data_source
        
    except Exception as e:
        print(f"Error in predict_colleges_for_rank: {e}")
        return [], False, None, None

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Handles the college prediction logic by fetching data dynamically.
    """
    results = []
    has_results = False
    data_year = None
    data_source = None
    error_message = None

    if request.method == "POST":
        try:
            rank = int(request.form.get("rank"))
            caste = request.form.get("caste")

            results, has_results, data_year, data_source = predict_colleges_for_rank(rank, caste)
            
            if not has_results:
                error_message = "No colleges found for your rank and caste combination. Try a different rank or check back later for updated data."

        except ValueError:
            error_message = "Please enter a valid rank number."
        except Exception as e:
            print(f"Error processing prediction request: {e}")
            error_message = f"An error occurred while processing your request. Please try again."
            
    return render_template("index.html", 
                         results=results, 
                         has_results=has_results,
                         data_year=data_year,
                         data_source=data_source,
                         error_message=error_message)

if __name__ == '__main__':
    # Get port from environment variable (for Render) or use default
    port = int(os.environ.get('PORT', 5001))
    
    # Run in debug mode only in development
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(host='0.0.0.0', port=port, debug=debug)

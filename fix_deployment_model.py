import joblib
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Path to the current model file
model_path = 'eamcet_rank_model_deployable.joblib'

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"❌ Model file not found: {model_path}")
    exit(1)

# Load the model file
try:
    model = joblib.load(model_path)
    print(f"✅ Successfully loaded model from {model_path}")
    
    # Create a properly structured model data dictionary
    model_data = {
        'best_model': model,
        'best_model_name': 'random_forest',
        'version': 'deployable_v1.0',
        'timestamp': '2023-11-01T00:00:00',
        'feature_names': ['mathematics', 'physics', 'chemistry'],
        'scaler': None  # Add scaler field to prevent KeyError
    }
    
    # Save the properly structured model
    joblib.dump(model_data, 'eamcet_rank_model_deployable_fixed.joblib')
    print(f"✅ Fixed model saved to eamcet_rank_model_deployable_fixed.joblib")
    
    # Backup the original model
    os.rename('eamcet_rank_model_deployable.joblib', 'eamcet_rank_model_deployable_original.joblib')
    print(f"✅ Original model backed up to eamcet_rank_model_deployable_original.joblib")
    
    # Rename the fixed model to the original name
    os.rename('eamcet_rank_model_deployable_fixed.joblib', 'eamcet_rank_model_deployable.joblib')
    print(f"✅ Fixed model renamed to eamcet_rank_model_deployable.joblib")
    
    print("\n✅ Model fixed successfully! The model is now compatible with Render deployment.")
    
except Exception as e:
    print(f"❌ Error fixing model: {str(e)}")
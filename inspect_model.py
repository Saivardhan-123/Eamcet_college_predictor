import joblib
import os
import json

# Path to the model file
model_path = 'eamcet_rank_model_deployable.joblib'

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"❌ Model file not found: {model_path}")
    exit(1)

# Load the model file
try:
    model_data = joblib.load(model_path)
    print(f"✅ Successfully loaded model from {model_path}")
    
    # Print model structure
    print("\nModel Structure:")
    if isinstance(model_data, dict):
        print(f"Model is a dictionary with keys: {list(model_data.keys())}")
        
        # Check for best_model
        if 'best_model' in model_data:
            print(f"\nBest model type: {type(model_data['best_model'])}")
            
            # Check for feature_names_in_
            if hasattr(model_data['best_model'], 'feature_names_in_'):
                print(f"Feature names: {model_data['best_model'].feature_names_in_}")
            
        # Check for version
        if 'version' in model_data:
            print(f"Version: {model_data['version']}")
            
        # Check for feature_names
        if 'feature_names' in model_data:
            print(f"Feature names: {model_data['feature_names']}")
    else:
        print(f"Model is not a dictionary, but a {type(model_data)}")
        
        # If it's a direct model object
        if hasattr(model_data, 'feature_names_in_'):
            print(f"Feature names: {model_data.feature_names_in_}")
    
    # Try to make a simple prediction to test the model
    print("\nAttempting a test prediction...")
    
    # Create test data
    import pandas as pd
    import numpy as np
    
    # Test with different formats to see what works
    test_data_formats = [
        # Format 1: Simple array
        np.array([[70, 35, 32]]),
        
        # Format 2: DataFrame with basic columns
        pd.DataFrame([[70, 35, 32]], columns=['mathematics', 'physics', 'chemistry']),
        
        # Format 3: DataFrame with extended features
        pd.DataFrame([
            [70, 35, 32, 137, 85.6, 0.875, 33.5, 1.04]
        ], columns=[
            'mathematics', 'physics', 'chemistry', 'total_score', 'percentile',
            'math_ratio', 'science_avg', 'math_dominance'
        ])
    ]
    
    # Try each format
    for i, test_data in enumerate(test_data_formats):
        try:
            if isinstance(model_data, dict) and 'best_model' in model_data:
                prediction = model_data['best_model'].predict(test_data)
            else:
                prediction = model_data.predict(test_data)
            print(f"Format {i+1} worked! Prediction: {prediction}")
            break
        except Exception as e:
            print(f"Format {i+1} failed: {str(e)}")
    
except Exception as e:
    print(f"❌ Error inspecting model: {str(e)}")
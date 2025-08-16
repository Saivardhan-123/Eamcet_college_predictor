#!/usr/bin/env python3
"""
Train a deployable ML model for EAMCET rank prediction.
This creates a smaller model that can be included in the repository.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def train_deployable_model():
    """
    Train a smaller, deployable model for EAMCET rank prediction.
    """
    print("🎯 Training deployable ML model for EAMCET rank prediction...")
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 2000  # Smaller dataset for deployable model
    
    data = []
    for i in range(n_samples):
        # Generate realistic score distributions
        math_score = np.random.beta(2, 2) * 80
        math_score = max(0, min(80, math_score))
        
        physics_score = np.random.beta(2.2, 2) * 40
        physics_score = max(0, min(40, physics_score))
        
        chemistry_score = np.random.beta(2.1, 2) * 40
        chemistry_score = max(0, min(40, chemistry_score))
        
        total_score = math_score + physics_score + chemistry_score
        percentile = (total_score / 160) * 100
        percentile += np.random.normal(0, 5)
        percentile = max(5, min(99.5, percentile))
        
        # Convert to rank
        rank = int(150000 * (100 - percentile) / 100)
        rank = max(1, min(150000, rank))
        
        # Add subject-specific bonuses
        if math_score > 60:
            rank = max(1, rank - np.random.randint(500, 2000))
        if physics_score > 30:
            rank = max(1, rank - np.random.randint(100, 1000))
        if chemistry_score > 30:
            rank = max(1, rank - np.random.randint(100, 1000))
        
        data.append({
            'mathematics': round(math_score, 1),
            'physics': round(physics_score, 1),
            'chemistry': round(chemistry_score, 1),
            'total_score': round(total_score, 1),
            'rank': rank
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    print(f"📊 Generated {len(df)} training samples")
    
    # Prepare features and target
    X = df[['mathematics', 'physics', 'chemistry']]
    y = df['rank']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model (smaller Random Forest for deployment)
    model = RandomForestRegressor(
        n_estimators=50,  # Smaller number of trees
        max_depth=10,     # Limit depth
        random_state=42,
        n_jobs=-1
    )
    
    print("🤖 Training Random Forest model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"📈 Model Performance:")
    print(f"   Mean Absolute Error: {mae:.0f} ranks")
    print(f"   R² Score: {r2:.3f}")
    
    # Save model
    model_filename = 'eamcet_rank_model_deployable.joblib'
    joblib.dump(model, model_filename)
    
    # Check file size
    file_size = os.path.getsize(model_filename) / (1024 * 1024)  # MB
    print(f"💾 Model saved as '{model_filename}' ({file_size:.1f} MB)")
    
    # Test model
    test_input = np.array([[70, 35, 35]])  # Example scores
    predicted_rank = model.predict(test_input)[0]
    print(f"🧪 Test prediction: Math=70, Physics=35, Chemistry=35 → Rank: {predicted_rank:.0f}")
    
    print("✅ Deployable model training completed!")
    print("📝 Add this model to your repository for Render deployment")
    
    return model_filename

if __name__ == "__main__":
    train_deployable_model()

#!/usr/bin/env python3
"""
EAMCET Rank Predictor using Machine Learning
Predicts EAMCET rank based on subject-wise scores using various ML algorithms.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime
import json

class EamcetRankPredictor:
    """
    Machine Learning model to predict EAMCET rank based on subject scores.
    """
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.is_trained = False
        
    def generate_synthetic_data(self, n_samples=5000):
        """
        Generate synthetic EAMCET data based on realistic patterns.
        
        EAMCET Pattern:
        - Mathematics: 80 marks (40 questions × 2 marks each)
        - Physics: 40 marks (40 questions × 1 mark each) 
        - Chemistry: 40 marks (40 questions × 1 mark each)
        - Total: 160 marks
        """
        np.random.seed(42)
        
        data = []
        
        for i in range(n_samples):
            # Generate realistic score distributions
            # Higher scores should generally lead to better ranks
            
            # Mathematics (0-80): Most important subject
            math_score = np.random.beta(2, 2) * 80
            math_score = max(0, min(80, math_score))
            
            # Physics (0-40): Moderately difficult
            physics_score = np.random.beta(2.2, 2) * 40
            physics_score = max(0, min(40, physics_score))
            
            # Chemistry (0-40): Similar to physics
            chemistry_score = np.random.beta(2.1, 2) * 40
            chemistry_score = max(0, min(40, chemistry_score))
            
            # Total score
            total_score = math_score + physics_score + chemistry_score
            
            # Calculate percentile (higher score = higher percentile)
            percentile = (total_score / 160) * 100
            
            # Add some randomness to make it realistic
            percentile += np.random.normal(0, 5)  # Add noise
            percentile = max(5, min(99.5, percentile))  # Clamp percentile
            
            # Convert percentile to rank (higher percentile = lower rank number)
            # Assuming total candidates ≈ 150,000
            rank = int(150000 * (100 - percentile) / 100)
            rank = max(1, min(150000, rank))
            
            # Add some subject-specific bonuses/penalties
            if math_score > 60:  # Good math score bonus
                rank = max(1, rank - np.random.randint(500, 2000))
            
            if physics_score > 30:  # Good physics bonus
                rank = max(1, rank - np.random.randint(100, 1000))
                
            if chemistry_score > 30:  # Good chemistry bonus
                rank = max(1, rank - np.random.randint(100, 1000))
            
            data.append({
                'mathematics': round(math_score, 1),
                'physics': round(physics_score, 1),
                'chemistry': round(chemistry_score, 1),
                'total_score': round(total_score, 1),
                'percentile': round(percentile, 2),
                'rank': rank
            })
        
        df = pd.DataFrame(data)
        
        # Add some categorical features
        df['math_grade'] = pd.cut(df['mathematics'], 
                                bins=[0, 20, 40, 60, 80], 
                                labels=['Poor', 'Average', 'Good', 'Excellent'])
        
        df['total_grade'] = pd.cut(df['total_score'], 
                                 bins=[0, 50, 100, 130, 160], 
                                 labels=['Poor', 'Average', 'Good', 'Excellent'])
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare feature matrix for training/prediction.
        """
        features = ['mathematics', 'physics', 'chemistry', 'total_score', 'percentile']
        
        X = df[features].copy()
        
        # Add engineered features
        X['math_ratio'] = X['mathematics'] / 80  # Math as ratio
        X['science_avg'] = (X['physics'] + X['chemistry']) / 2  # Average science score
        X['math_dominance'] = X['mathematics'] / (X['physics'] + X['chemistry'] + 1)  # Math vs science
        
        return X
    
    def train_models(self, df):
        """
        Train multiple ML models and select the best one.
        """
        print("🤖 Training ML models for rank prediction...")
        
        X = self.prepare_features(df)
        y = df['rank']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model_scores = {}
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            
            # Train model
            if name == 'linear_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate model
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            model_scores[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'model': model
            }
            
            print(f"    MAE: {mae:.0f}, RMSE: {rmse:.0f}, R²: {r2:.3f}")
        
        # Select best model based on lowest MAE
        best_name = min(model_scores.keys(), key=lambda k: model_scores[k]['mae'])
        self.best_model = model_scores[best_name]['model']
        self.best_model_name = best_name
        self.is_trained = True
        
        print(f"✅ Best model: {best_name} (MAE: {model_scores[best_name]['mae']:.0f})")
        
        return model_scores
    
    def predict_rank(self, math_score, physics_score, chemistry_score):
        """
        Predict EAMCET rank based on subject scores.
        
        Args:
            math_score (float): Mathematics score (0-80)
            physics_score (float): Physics score (0-40)
            chemistry_score (float): Chemistry score (0-40)
            
        Returns:
            dict: Prediction results with confidence metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train_models() first.")
        
        # Validate inputs
        if not (0 <= math_score <= 80):
            raise ValueError("Mathematics score must be between 0 and 80")
        if not (0 <= physics_score <= 40):
            raise ValueError("Physics score must be between 0 and 40") 
        if not (0 <= chemistry_score <= 40):
            raise ValueError("Chemistry score must be between 0 and 40")
        
        # Calculate derived features
        total_score = math_score + physics_score + chemistry_score
        percentile = (total_score / 160) * 100
        
        # Handle specialized model (simpler feature set)
        if self.best_model_name == 'specialized_random_forest':
            features = [[math_score, physics_score, chemistry_score, total_score]]
            predicted_rank = self.best_model.predict(features)[0]
        else:
            # Standard model with full feature engineering
            features = np.array([[
                math_score, physics_score, chemistry_score, total_score, percentile,
                math_score / 80,  # math_ratio
                (physics_score + chemistry_score) / 2,  # science_avg
                math_score / (physics_score + chemistry_score + 1)  # math_dominance
            ]])
            
            # Make prediction
            if self.best_model_name == 'linear_regression':
                features_scaled = self.scaler.transform(features)
                predicted_rank = self.best_model.predict(features_scaled)[0]
            else:
                predicted_rank = self.best_model.predict(features)[0]
        
        # Handle deployable model (uses feature names)
        if hasattr(self.best_model, 'feature_names_in_'):
            # Create DataFrame with proper feature names
            import pandas as pd
            features_df = pd.DataFrame([[math_score, physics_score, chemistry_score]], 
                                     columns=['mathematics', 'physics', 'chemistry'])
            predicted_rank = self.best_model.predict(features_df)[0]
        
        # Ensure rank is within valid range
        predicted_rank = max(1, min(150000, int(predicted_rank)))
        
        # Calculate confidence metrics
        confidence = self._calculate_confidence(total_score, predicted_rank)
        
        return {
            'predicted_rank': predicted_rank,
            'total_score': total_score,
            'percentile': round(percentile, 2),
            'confidence': confidence,
            'model_used': self.best_model_name,
            'rank_range': self._get_rank_range(predicted_rank),
            'performance_level': self._get_performance_level(percentile)
        }
    
    def _calculate_confidence(self, total_score, predicted_rank):
        """Calculate prediction confidence based on score patterns."""
        # Higher scores generally have higher confidence
        score_confidence = min(100, (total_score / 160) * 100 + 20)
        
        # Mid-range ranks have higher confidence than extreme ranks
        rank_factor = 1.0
        if predicted_rank < 1000 or predicted_rank > 100000:
            rank_factor = 0.8
        
        confidence = score_confidence * rank_factor
        return round(min(95, confidence), 1)  # Cap at 95%
    
    def _get_rank_range(self, predicted_rank):
        """Get likely rank range based on prediction."""
        margin = max(500, predicted_rank * 0.1)  # 10% margin or 500, whichever is higher
        lower = max(1, int(predicted_rank - margin))
        upper = min(150000, int(predicted_rank + margin))
        return {'lower': lower, 'upper': upper}
    
    def _get_performance_level(self, percentile):
        """Categorize performance level based on percentile."""
        if percentile >= 95:
            return "Excellent (Top 5%)"
        elif percentile >= 85:
            return "Very Good (Top 15%)"
        elif percentile >= 70:
            return "Good (Top 30%)"
        elif percentile >= 50:
            return "Average (Top 50%)"
        else:
            return "Below Average"
    
    def save_model(self, filepath):
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        # Check if this is a specialized model
        if 'version' in model_data and model_data['version'] == 'specialized_v1.0':
            # Load specialized model
            self.best_model = model_data['model']
            self.best_model_name = 'specialized_random_forest'
            self.scaler = None  # Specialized model doesn't use scaler
            self.feature_names = model_data.get('feature_names', ['mathematics', 'physics', 'chemistry', 'total_score'])
        else:
            # Load standard model
            self.best_model = model_data['best_model']
            self.best_model_name = model_data['best_model_name']
            self.scaler = model_data['scaler']
            self.feature_names = model_data.get('feature_names', None)
        
        self.is_trained = True
        
        print(f"✅ Model loaded from {filepath}")
        print(f"   Model type: {self.best_model_name}")
        print(f"   Version: {model_data.get('version', 'v1.0')}")
        print(f"   Trained on: {model_data.get('timestamp', 'Unknown')}")

def main():
    """
    Main function to train and test the rank prediction model.
    """
    print("🎓 EAMCET Rank Prediction Model")
    print("=" * 50)
    
    # Initialize predictor
    predictor = EamcetRankPredictor()
    
    # Generate synthetic training data
    print("📊 Generating synthetic training data...")
    df = predictor.generate_synthetic_data(n_samples=10000)
    print(f"   Generated {len(df)} training samples")
    print(f"   Score range: {df['total_score'].min():.1f} - {df['total_score'].max():.1f}")
    print(f"   Rank range: {df['rank'].min()} - {df['rank'].max()}")
    
    # Train models
    model_scores = predictor.train_models(df)
    
    # Save the model
    model_path = 'eamcet_rank_model.joblib'
    predictor.save_model(model_path)
    
    # Test predictions
    print("\n🧪 Testing predictions...")
    test_cases = [
        (70, 35, 32),  # High scores
        (50, 25, 28),  # Average scores
        (30, 15, 18),  # Lower scores
        (80, 40, 40),  # Perfect scores
    ]
    
    for math, phy, chem in test_cases:
        result = predictor.predict_rank(math, phy, chem)
        print(f"\n📋 Scores: Math={math}, Physics={phy}, Chemistry={chem}")
        print(f"   🎯 Predicted Rank: {result['predicted_rank']:,}")
        print(f"   📊 Total Score: {result['total_score']}/160 ({result['percentile']}%)")
        print(f"   🎭 Performance: {result['performance_level']}")
        print(f"   📈 Confidence: {result['confidence']}%")
        print(f"   📍 Rank Range: {result['rank_range']['lower']:,} - {result['rank_range']['upper']:,}")
    
    print(f"\n✅ Model training complete!")
    print(f"📁 Model saved as: {model_path}")
    return predictor

if __name__ == "__main__":
    main()

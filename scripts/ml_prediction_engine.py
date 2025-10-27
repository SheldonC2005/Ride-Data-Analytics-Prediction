"""
Machine Learning Prediction Engine for Rapido Transportation
Author: BI Project Team
Date: October 14, 2025

This module implements comprehensive ML algorithms for:
1. Demand Forecasting (RandomForest + Linear Regression)
2. Dynamic Pricing Optimization (XGBoost + Gradient Boosting)
3. Route Recommendation (Decision Tree + Random Forest)
4. Weather Impact Prediction (Ridge Regression + XGBoost)

Features:
- Real-time prediction capabilities
- Model persistence and reloading
- Comprehensive evaluation metrics
- Feature engineering with weather and geospatial data
- Power BI integration ready
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                           accuracy_score, classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

# Geospatial and Weather
from geopy.distance import geodesic
import requests
import os
from datetime import datetime, timedelta
import json

# Model Persistence
import joblib
import pickle
from pathlib import Path

# Optimization
import optuna
from optuna.samplers import TPESampler

class RapidoMLEngine:
    """
    Comprehensive Machine Learning Engine for Rapido Transportation Analytics
    """
    
    def __init__(self, data_path, weather_api_key=None):
        """Initialize ML Engine"""
        self.data_path = data_path
        self.weather_api_key = weather_api_key
        self.df = None
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_metrics = {}
        
        # Create models directory
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        print("🤖 Rapido ML Engine Initialized")
    
    def load_and_prepare_data(self):
        """Load and prepare data with advanced feature engineering"""
        print("📊 Loading and preparing data for ML...")
        
        self.df = pd.read_csv(self.data_path)
        
        # Convert datetime columns
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Create route column
        self.df['route'] = self.df['from_location'] + ' → ' + self.df['to_location']
        
        print("✅ Data loaded successfully")
        return self._engineer_features()
    
    def _engineer_features(self):
        """Advanced feature engineering for ML models"""
        print("⚙️ Engineering advanced features...")
        
        # Time-based features (only create if not already present)
        if 'hour' not in self.df.columns:
            self.df['hour'] = self.df['datetime'].dt.hour
        if 'day_of_week' not in self.df.columns:
            self.df['day_of_week'] = self.df['datetime'].dt.dayofweek
        if 'month' not in self.df.columns:
            self.df['month'] = self.df['datetime'].dt.month
        if 'quarter' not in self.df.columns:
            self.df['quarter'] = self.df['datetime'].dt.quarter
        if 'is_weekend' not in self.df.columns:
            self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        
        # Rush hour features
        self.df['is_morning_rush'] = ((self.df['hour'] >= 8) & (self.df['hour'] <= 9)).astype(int)
        self.df['is_evening_rush'] = ((self.df['hour'] >= 17) & (self.df['hour'] <= 18)).astype(int)
        self.df['is_rush_hour'] = (self.df['is_morning_rush'] | self.df['is_evening_rush']).astype(int)
        
        # Geospatial features
        self._add_geospatial_features()
        
        # Weather features
        self._add_weather_features()
        
        # Historical demand features
        self._add_historical_features()
        
        # Route and location features
        self._add_route_features()
        
        print("✅ Feature engineering completed")
        return self.df
    
    def _add_geospatial_features(self):
        """Add geospatial features based on locations"""
        print("📍 Adding geospatial features...")
        
        # VIT University coordinates (approximate)
        vit_coords = (12.9685, 79.1550)
        
        # Check if route involves VIT
        self.df['involves_vit'] = (
            self.df['from_location'].str.contains('VIT', case=False) | 
            self.df['to_location'].str.contains('VIT', case=False)
        ).astype(int)
        
        # Route complexity (based on common locations)
        high_traffic_locations = ['VIT', 'Railway', 'CMC', 'Green Circle', 'Katpadi']
        self.df['route_complexity'] = 0
        
        for location in high_traffic_locations:
            self.df['route_complexity'] += (
                self.df['from_location'].str.contains(location, case=False) |
                self.df['to_location'].str.contains(location, case=False)
            ).astype(int)
        
        # Distance categories
        self.df['distance_category'] = pd.cut(
            self.df['distance_km'], 
            bins=[0, 2, 5, 10, 20, float('inf')], 
            labels=['very_short', 'short', 'medium', 'long', 'very_long']
        )
        
        # Route frequency (how often this route is taken)
        route_counts = self.df.groupby('route').size()
        self.df['route_frequency'] = self.df['route'].map(route_counts)
        
    def _add_weather_features(self):
        """Add weather-related features"""
        print("🌤️ Adding weather features...")
        
        # Weather impact scores
        weather_impact = {
            'Clear': 1.0,
            'Clouds': 1.1,
            'Haze': 1.3,
            'Rain': 1.9,
            'Thunderstorm': 2.3,
            'Drizzle': 1.6,
            'Mist': 1.2
        }
        
        self.df['weather_impact_score'] = self.df['weather_condition'].map(weather_impact).fillna(1.0)
        
        # Seasonal patterns
        self.df['season'] = self.df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'monsoon', 10: 'monsoon', 11: 'winter'
        })
        
        # Rain prediction (based on historical patterns)
        self.df['rain_probability'] = self.df.groupby(['month', 'hour'])['weather_condition'].transform(
            lambda x: (x == 'Rain').mean()
        )
        
    def _add_historical_features(self):
        """Add historical demand and pricing features"""
        print("📈 Adding historical features...")
        
        # Sort by datetime for rolling calculations
        self.df = self.df.sort_values('datetime')
        
        # Historical demand features (rolling averages)
        self.df['hourly_demand_avg'] = self.df.groupby('hour').size().reindex(self.df['hour']).values
        
        # Previous hour demand
        hourly_counts = self.df.groupby(['date', 'hour']).size().reset_index(name='hourly_count')
        hourly_counts['prev_hour_demand'] = hourly_counts.groupby('date')['hourly_count'].shift(1)
        
        # Merge back (only prev_hour_demand to avoid duplicate columns)
        self.df = self.df.merge(
            hourly_counts[['date', 'hour', 'prev_hour_demand']], 
            on=['date', 'hour'], 
            how='left',
            suffixes=('', '_dup')
        )
        # Remove any duplicate columns created during merge
        dup_cols = [col for col in self.df.columns if col.endswith('_dup')]
        if dup_cols:
            self.df.drop(columns=dup_cols, inplace=True)
        self.df['prev_hour_demand'].fillna(0, inplace=True)
        
        # Price volatility
        self.df['price_volatility'] = self.df.groupby(['date'])['final_price'].transform('std').fillna(0)
        
    def _add_route_features(self):
        """Add route-specific features"""
        print("🛣️ Adding route features...")
        
        # Route performance metrics
        route_stats = self.df.groupby('route').agg({
            'final_price': ['mean', 'std'],
            'waiting_time_minutes': 'mean',
            'surge_multiplier': 'mean'
        }).round(2)
        
        route_stats.columns = ['route_avg_price', 'route_price_std', 'route_avg_wait', 'route_avg_surge']
        route_stats['route_price_std'].fillna(0, inplace=True)
        
        # Merge back to main dataframe
        self.df = self.df.merge(route_stats, left_on='route', right_index=True, how='left')
        
    def prepare_ml_features(self):
        """Prepare features for ML models"""
        print("🔧 Preparing ML features...")
        
        # Encode categorical variables
        categorical_cols = ['vehicle_type', 'weather_condition', 'season', 'distance_category']
        
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                self.df[f'{col}_encoded'] = self.encoders[col].fit_transform(self.df[col].astype(str))
            else:
                self.df[f'{col}_encoded'] = self.encoders[col].transform(self.df[col].astype(str))
        
        # Select features for different models
        self.demand_features = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
            'weather_impact_score', 'involves_vit', 'route_complexity', 
            'prev_hour_demand', 'rain_probability', 'season_encoded'
        ]
        
        self.pricing_features = [
            'hour', 'day_of_week', 'distance_km', 'weather_condition_encoded',
            'vehicle_type_encoded', 'is_rush_hour', 'weather_impact_score',
            'route_frequency', 'route_avg_price', 'price_volatility'
        ]
        
        self.route_features = [
            'hour', 'distance_km', 'weather_condition_encoded', 'is_rush_hour',
            'route_complexity', 'involves_vit', 'route_avg_wait'
        ]
        
        self.weather_impact_features = [
            'hour', 'month', 'weather_condition_encoded', 'season_encoded',
            'rain_probability', 'distance_km', 'is_rush_hour'
        ]
        
        print("✅ ML features prepared")
    
    # ========== DEMAND FORECASTING ==========
    
    def train_demand_forecasting_models(self):
        """Train demand forecasting models"""
        print("\n🔮 Training Demand Forecasting Models...")
        
        # Prepare demand data (trips per hour)
        demand_data = self.df.groupby(['date', 'hour']).size().reset_index(name='demand')
        
        # Get unique features (avoid duplicates from merge keys)
        features_to_merge = ['date', 'hour'] + [f for f in self.demand_features if f not in ['date', 'hour']]
        demand_data = demand_data.merge(
            self.df[features_to_merge].drop_duplicates(),
            on=['date', 'hour'],
            how='left'
        )
        
        # Remove any rows with NaN
        demand_data = demand_data.dropna()
        
        X = demand_data[self.demand_features]
        y = demand_data['demand']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scalers['demand'] = StandardScaler()
        X_train_scaled = self.scalers['demand'].fit_transform(X_train)
        X_test_scaled = self.scalers['demand'].transform(X_test)
        
        # Train RandomForest
        print("Training RandomForest for demand...")
        rf_demand = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_demand.fit(X_train_scaled, y_train)
        
        # Train Linear Regression
        print("Training Linear Regression for demand...")
        lr_demand = LinearRegression()
        lr_demand.fit(X_train_scaled, y_train)
        
        # Evaluate models
        rf_pred = rf_demand.predict(X_test_scaled)
        lr_pred = lr_demand.predict(X_test_scaled)
        
        # Store models and metrics
        self.models['demand_rf'] = rf_demand
        self.models['demand_lr'] = lr_demand
        
        self.model_metrics['demand_forecasting'] = {
            'RandomForest': {
                'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
                'MAE': mean_absolute_error(y_test, rf_pred),
                'R2': r2_score(y_test, rf_pred)
            },
            'LinearRegression': {
                'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
                'MAE': mean_absolute_error(y_test, lr_pred),
                'R2': r2_score(y_test, lr_pred)
            }
        }
        
        print("✅ Demand forecasting models trained")
        self._print_model_metrics('demand_forecasting')
    
    # ========== DYNAMIC PRICING ==========
    
    def train_pricing_models(self):
        """Train dynamic pricing optimization models"""
        print("\n💰 Training Dynamic Pricing Models...")
        
        # Prepare pricing data
        pricing_data = self.df[self.pricing_features + ['surge_multiplier']].dropna()
        
        X = pricing_data[self.pricing_features]
        y = pricing_data['surge_multiplier']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scalers['pricing'] = StandardScaler()
        X_train_scaled = self.scalers['pricing'].fit_transform(X_train)
        X_test_scaled = self.scalers['pricing'].transform(X_test)
        
        # Train XGBoost
        print("Training XGBoost for pricing...")
        xgb_pricing = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        xgb_pricing.fit(X_train_scaled, y_train)
        
        # Train Gradient Boosting
        print("Training Gradient Boosting for pricing...")
        gb_pricing = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        gb_pricing.fit(X_train_scaled, y_train)
        
        # Evaluate models
        xgb_pred = xgb_pricing.predict(X_test_scaled)
        gb_pred = gb_pricing.predict(X_test_scaled)
        
        # Store models and metrics
        self.models['pricing_xgb'] = xgb_pricing
        self.models['pricing_gb'] = gb_pricing
        
        self.model_metrics['dynamic_pricing'] = {
            'XGBoost': {
                'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
                'MAE': mean_absolute_error(y_test, xgb_pred),
                'R2': r2_score(y_test, xgb_pred)
            },
            'GradientBoosting': {
                'RMSE': np.sqrt(mean_squared_error(y_test, gb_pred)),
                'MAE': mean_absolute_error(y_test, gb_pred),
                'R2': r2_score(y_test, gb_pred)
            }
        }
        
        print("✅ Pricing models trained")
        self._print_model_metrics('dynamic_pricing')
    
    # ========== ROUTE RECOMMENDATION ==========
    
    def train_route_recommendation_models(self):
        """Train route recommendation models"""
        print("\n🗺️ Training Route Recommendation Models...")
        
        # Prepare route data (predict optimal vehicle type)
        route_data = self.df[self.route_features + ['vehicle_type']].dropna()
        
        X = route_data[self.route_features]
        y = route_data['vehicle_type']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scalers['route'] = StandardScaler()
        X_train_scaled = self.scalers['route'].fit_transform(X_train)
        X_test_scaled = self.scalers['route'].transform(X_test)
        
        # Train Decision Tree
        print("Training Decision Tree for route recommendation...")
        dt_route = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            random_state=42
        )
        dt_route.fit(X_train_scaled, y_train)
        
        # Train Random Forest
        print("Training Random Forest for route recommendation...")
        rf_route = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_route.fit(X_train_scaled, y_train)
        
        # Evaluate models
        dt_pred = dt_route.predict(X_test_scaled)
        rf_pred = rf_route.predict(X_test_scaled)
        
        # Store models and metrics
        self.models['route_dt'] = dt_route
        self.models['route_rf'] = rf_route
        
        self.model_metrics['route_recommendation'] = {
            'DecisionTree': {
                'Accuracy': accuracy_score(y_test, dt_pred),
                'Classification_Report': classification_report(y_test, dt_pred, output_dict=True)
            },
            'RandomForest': {
                'Accuracy': accuracy_score(y_test, rf_pred),
                'Classification_Report': classification_report(y_test, rf_pred, output_dict=True)
            }
        }
        
        print("✅ Route recommendation models trained")
        self._print_model_metrics('route_recommendation')
    
    # ========== WEATHER IMPACT PREDICTION ==========
    
    def train_weather_impact_models(self):
        """Train weather impact prediction models"""
        print("\n🌦️ Training Weather Impact Models...")
        
        # Prepare weather impact data (predict price premium based on weather)
        weather_data = self.df[self.weather_impact_features + ['price_premium_percent']].dropna()
        
        X = weather_data[self.weather_impact_features]
        y = weather_data['price_premium_percent']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scalers['weather'] = StandardScaler()
        X_train_scaled = self.scalers['weather'].fit_transform(X_train)
        X_test_scaled = self.scalers['weather'].transform(X_test)
        
        # Train Ridge Regression
        print("Training Ridge Regression for weather impact...")
        ridge_weather = Ridge(alpha=1.0)
        ridge_weather.fit(X_train_scaled, y_train)
        
        # Train XGBoost
        print("Training XGBoost for weather impact...")
        xgb_weather = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        xgb_weather.fit(X_train_scaled, y_train)
        
        # Evaluate models
        ridge_pred = ridge_weather.predict(X_test_scaled)
        xgb_pred = xgb_weather.predict(X_test_scaled)
        
        # Store models and metrics
        self.models['weather_ridge'] = ridge_weather
        self.models['weather_xgb'] = xgb_weather
        
        self.model_metrics['weather_impact'] = {
            'Ridge': {
                'RMSE': np.sqrt(mean_squared_error(y_test, ridge_pred)),
                'MAE': mean_absolute_error(y_test, ridge_pred),
                'R2': r2_score(y_test, ridge_pred)
            },
            'XGBoost': {
                'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
                'MAE': mean_absolute_error(y_test, xgb_pred),
                'R2': r2_score(y_test, xgb_pred)
            }
        }
        
        print("✅ Weather impact models trained")
        self._print_model_metrics('weather_impact')
    
    # ========== MODEL PERSISTENCE ==========
    
    def save_models(self):
        """Save all trained models and scalers"""
        print("\n💾 Saving models...")
        
        # Save models
        for name, model in self.models.items():
            model_path = self.models_dir / f"{name}.joblib"
            joblib.dump(model, model_path)
            print(f"Saved {name} to {model_path}")
        
        # Save scalers
        scalers_path = self.models_dir / "scalers.joblib"
        joblib.dump(self.scalers, scalers_path)
        
        # Save encoders
        encoders_path = self.models_dir / "encoders.joblib"
        joblib.dump(self.encoders, encoders_path)
        
        # Save metrics
        metrics_path = self.models_dir / "model_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.model_metrics, f, indent=2)
        
        print("✅ All models saved successfully")
    
    def load_models(self):
        """Load saved models and scalers
        
        Returns:
            bool: True if models loaded successfully, False if no models found
        """
        print("📁 Loading saved models...")
        
        # Check if any model files exist
        model_files = list(self.models_dir.glob("*.joblib"))
        if not model_files:
            return False
        
        # Load models
        for model_file in model_files:
            if model_file.stem not in ['scalers', 'encoders']:
                model = joblib.load(model_file)
                self.models[model_file.stem] = model
                print(f"Loaded {model_file.stem}")
        
        # Load scalers
        scalers_path = self.models_dir / "scalers.joblib"
        if scalers_path.exists():
            self.scalers = joblib.load(scalers_path)
        
        # Load encoders
        encoders_path = self.models_dir / "encoders.joblib"
        if encoders_path.exists():
            self.encoders = joblib.load(encoders_path)
        
        # Load metrics
        metrics_path = self.models_dir / "model_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                self.model_metrics = json.load(f)
        
        print("✅ Models loaded successfully")
        return True
    
    # ========== REAL-TIME PREDICTIONS ==========
    
    def predict_demand(self, hour, day_of_week, month, is_weekend, weather_condition, **kwargs):
        """Real-time demand prediction"""
        if 'demand_rf' not in self.models:
            raise ValueError("Demand model not trained. Call train_demand_forecasting_models() first.")
        
        # Prepare input features
        weather_impact = kwargs.get('weather_impact_score', 1.0)
        involves_vit = kwargs.get('involves_vit', 0)
        route_complexity = kwargs.get('route_complexity', 1)
        
        features = np.array([[
            hour, day_of_week, month, int(is_weekend), 
            int((8 <= hour <= 9) or (17 <= hour <= 18)),  # is_rush_hour
            weather_impact, involves_vit, route_complexity,
            kwargs.get('prev_hour_demand', 0),
            kwargs.get('rain_probability', 0.1),
            kwargs.get('season_encoded', 1)
        ]])
        
        # Scale features
        features_scaled = self.scalers['demand'].transform(features)
        
        # Predict
        rf_prediction = self.models['demand_rf'].predict(features_scaled)[0]
        lr_prediction = self.models['demand_lr'].predict(features_scaled)[0]
        
        return {
            'RandomForest': max(0, int(rf_prediction)),
            'LinearRegression': max(0, int(lr_prediction)),
            'Ensemble': max(0, int((rf_prediction + lr_prediction) / 2))
        }
    
    def predict_pricing(self, hour, day_of_week, distance_km, weather_condition, vehicle_type, **kwargs):
        """Real-time pricing prediction"""
        if 'pricing_xgb' not in self.models:
            raise ValueError("Pricing model not trained. Call train_pricing_models() first.")
        
        # Encode categorical variables
        weather_encoded = self.encoders['weather_condition'].transform([weather_condition])[0]
        vehicle_encoded = self.encoders['vehicle_type'].transform([vehicle_type])[0]
        
        features = np.array([[
            hour, day_of_week, distance_km, weather_encoded, vehicle_encoded,
            int((8 <= hour <= 9) or (17 <= hour <= 18)),  # is_rush_hour
            kwargs.get('weather_impact_score', 1.0),
            kwargs.get('route_frequency', 100),
            kwargs.get('route_avg_price', 150),
            kwargs.get('price_volatility', 20)
        ]])
        
        # Scale features
        features_scaled = self.scalers['pricing'].transform(features)
        
        # Predict
        xgb_prediction = self.models['pricing_xgb'].predict(features_scaled)[0]
        gb_prediction = self.models['pricing_gb'].predict(features_scaled)[0]
        
        return {
            'XGBoost': max(1.0, xgb_prediction),
            'GradientBoosting': max(1.0, gb_prediction),
            'Ensemble': max(1.0, (xgb_prediction + gb_prediction) / 2)
        }
    
    def recommend_vehicle(self, hour, distance_km, weather_condition, **kwargs):
        """Real-time vehicle recommendation"""
        if 'route_rf' not in self.models:
            raise ValueError("Route model not trained. Call train_route_recommendation_models() first.")
        
        # Encode weather
        weather_encoded = self.encoders['weather_condition'].transform([weather_condition])[0]
        
        features = np.array([[
            hour, distance_km, weather_encoded,
            int((8 <= hour <= 9) or (17 <= hour <= 18)),  # is_rush_hour
            kwargs.get('route_complexity', 1),
            kwargs.get('involves_vit', 0),
            kwargs.get('route_avg_wait', 4.0)
        ]])
        
        # Scale features
        features_scaled = self.scalers['route'].transform(features)
        
        # Predict
        dt_prediction = self.models['route_dt'].predict(features_scaled)[0]
        rf_prediction = self.models['route_rf'].predict(features_scaled)[0]
        
        # Get probabilities
        dt_proba = self.models['route_dt'].predict_proba(features_scaled)[0]
        rf_proba = self.models['route_rf'].predict_proba(features_scaled)[0]
        
        return {
            'DecisionTree': dt_prediction,
            'RandomForest': rf_prediction,
            'Probabilities': {
                'DecisionTree': dict(zip(self.models['route_dt'].classes_, dt_proba)),
                'RandomForest': dict(zip(self.models['route_rf'].classes_, rf_proba))
            }
        }
    
    def predict_weather_impact(self, hour, month, weather_condition, **kwargs):
        """Real-time weather impact prediction"""
        if 'weather_xgb' not in self.models:
            raise ValueError("Weather model not trained. Call train_weather_impact_models() first.")
        
        # Encode categorical variables
        weather_encoded = self.encoders['weather_condition'].transform([weather_condition])[0]
        season_encoded = kwargs.get('season_encoded', 1)
        
        features = np.array([[
            hour, month, weather_encoded, season_encoded,
            kwargs.get('rain_probability', 0.1),
            kwargs.get('distance_km', 5.0),
            int((8 <= hour <= 9) or (17 <= hour <= 18))  # is_rush_hour
        ]])
        
        # Scale features
        features_scaled = self.scalers['weather'].transform(features)
        
        # Predict
        ridge_prediction = self.models['weather_ridge'].predict(features_scaled)[0]
        xgb_prediction = self.models['weather_xgb'].predict(features_scaled)[0]
        
        return {
            'Ridge': ridge_prediction,
            'XGBoost': xgb_prediction,
            'Ensemble': (ridge_prediction + xgb_prediction) / 2
        }
    
    # ========== BATCH PREDICTIONS FOR POWER BI ==========
    
    def generate_batch_predictions(self):
        """Generate batch predictions for Power BI integration"""
        print("\n📊 Generating batch predictions for Power BI...")
        
        # Create prediction scenarios
        hours = range(24)
        days = range(7)  # 0=Monday, 6=Sunday
        weather_conditions = ['Clear', 'Rain', 'Clouds', 'Thunderstorm']
        vehicle_types = ['auto', 'bike', 'cab']
        
        predictions = []
        
        for hour in hours:
            for day in days:
                for weather in weather_conditions:
                    # Demand prediction
                    try:
                        demand_pred = self.predict_demand(
                            hour=hour, 
                            day_of_week=day, 
                            month=6,  # Average month
                            is_weekend=(day >= 5),
                            weather_condition=weather
                        )
                    except:
                        demand_pred = {'Ensemble': 0}
                    
                    for vehicle in vehicle_types:
                        # Pricing prediction
                        try:
                            pricing_pred = self.predict_pricing(
                                hour=hour,
                                day_of_week=day,
                                distance_km=5.0,  # Average distance
                                weather_condition=weather,
                                vehicle_type=vehicle
                            )
                        except:
                            pricing_pred = {'Ensemble': 1.5}
                        
                        # Vehicle recommendation
                        try:
                            vehicle_rec = self.recommend_vehicle(
                                hour=hour,
                                distance_km=5.0,
                                weather_condition=weather
                            )
                        except:
                            vehicle_rec = {'RandomForest': vehicle}
                        
                        # Weather impact
                        try:
                            weather_impact = self.predict_weather_impact(
                                hour=hour,
                                month=6,
                                weather_condition=weather
                            )
                        except:
                            weather_impact = {'Ensemble': 0}
                        
                        predictions.append({
                            'hour': hour,
                            'day_of_week': day,
                            'weather_condition': weather,
                            'vehicle_type': vehicle,
                            'predicted_demand': demand_pred['Ensemble'],
                            'predicted_surge': pricing_pred['Ensemble'],
                            'recommended_vehicle': vehicle_rec['RandomForest'],
                            'weather_impact_score': weather_impact['Ensemble'],
                            'is_weekend': int(day >= 5),
                            'is_rush_hour': int((8 <= hour <= 9) or (17 <= hour <= 18))
                        })
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Save for Power BI
        output_path = "power_bi/data_model/ml_predictions.xlsx"
        predictions_df.to_excel(output_path, index=False, sheet_name="ML_Predictions")
        
        print(f"✅ Batch predictions saved to {output_path}")
        return predictions_df
    
    # ========== UTILITY METHODS ==========
    
    def _print_model_metrics(self, model_type):
        """Print model evaluation metrics"""
        metrics = self.model_metrics[model_type]
        print(f"\n📈 {model_type.upper()} MODEL METRICS:")
        print("="*50)
        
        for model_name, model_metrics in metrics.items():
            print(f"\n{model_name}:")
            for metric, value in model_metrics.items():
                if isinstance(value, dict):
                    print(f"  {metric}:")
                    for sub_metric, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            print(f"    {sub_metric}: {sub_value}")
                        else:
                            print(f"    {sub_metric}: {sub_value:.4f}")
                else:
                    print(f"  {metric}: {value:.4f}")
    
    def get_model_summary(self):
        """Get comprehensive model summary"""
        summary = {
            'models_trained': len(self.models),
            'model_types': list(self.model_metrics.keys()),
            'best_models': {},
            'feature_counts': {
                'demand_features': len(self.demand_features),
                'pricing_features': len(self.pricing_features),
                'route_features': len(self.route_features),
                'weather_features': len(self.weather_impact_features)
            }
        }
        
        # Identify best models by metric
        for model_type, metrics in self.model_metrics.items():
            if model_type in ['demand_forecasting', 'dynamic_pricing', 'weather_impact']:
                best_r2 = 0
                best_model = None
                for model_name, model_metrics in metrics.items():
                    if 'R2' in model_metrics and model_metrics['R2'] > best_r2:
                        best_r2 = model_metrics['R2']
                        best_model = model_name
                summary['best_models'][model_type] = {'model': best_model, 'R2': best_r2}
            
            elif model_type == 'route_recommendation':
                best_accuracy = 0
                best_model = None
                for model_name, model_metrics in metrics.items():
                    if model_metrics['Accuracy'] > best_accuracy:
                        best_accuracy = model_metrics['Accuracy']
                        best_model = model_name
                summary['best_models'][model_type] = {'model': best_model, 'Accuracy': best_accuracy}
        
        return summary

# Example usage and testing
def main():
    """Main function for testing ML engine"""
    print("🚀 Initializing Rapido ML Engine...")
    
    # Initialize ML engine
    ml_engine = RapidoMLEngine("Dataset.csv")
    
    # Load and prepare data
    df = ml_engine.load_and_prepare_data()
    ml_engine.prepare_ml_features()
    
    # Train all models
    ml_engine.train_demand_forecasting_models()
    ml_engine.train_pricing_models()
    ml_engine.train_route_recommendation_models()
    ml_engine.train_weather_impact_models()
    
    # Save models
    ml_engine.save_models()
    
    # Generate predictions for Power BI
    predictions_df = ml_engine.generate_batch_predictions()
    
    # Print summary
    summary = ml_engine.get_model_summary()
    print("\n🎯 ML ENGINE SUMMARY:")
    print("="*50)
    print(json.dumps(summary, indent=2))
    
    # Test real-time predictions
    print("\n🔮 Testing Real-time Predictions...")
    
    # Test demand prediction
    demand_test = ml_engine.predict_demand(
        hour=18, day_of_week=1, month=6, 
        is_weekend=False, weather_condition='Rain'
    )
    print(f"Demand prediction for 6 PM on rainy weekday: {demand_test}")
    
    # Test pricing prediction
    pricing_test = ml_engine.predict_pricing(
        hour=18, day_of_week=1, distance_km=8.0,
        weather_condition='Rain', vehicle_type='auto'
    )
    print(f"Pricing prediction for 8km auto ride in rain: {pricing_test}")
    
    print("\n✅ ML Engine setup complete!")

if __name__ == "__main__":
    main()
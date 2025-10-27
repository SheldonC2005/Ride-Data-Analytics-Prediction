"""
Rapido Price Prediction Web Server
Provides web interface and API for ML-based ride price predictions

Usage: python prediction_server.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
sys.path.append('scripts')

from ml_prediction_engine import RapidoMLEngine  # type: ignore
import warnings
warnings.filterwarnings('ignore')
import socket
import numpy as np

app = Flask(__name__)


def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj

# Global ML engine
ml_engine = None


def find_free_port(start_port=5000, max_attempts=10):
    """Find an available port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            continue
    return start_port


def load_models():
    """Load trained ML models"""
    global ml_engine
    print("🔄 Loading trained ML models...")
    
    ml_engine = RapidoMLEngine("Dataset.csv")
    
    # Check if models exist
    if not ml_engine.load_models():
        print("\n❌ ERROR: No trained models found!")
        print("\n📝 Please run the training script first:")
        print("   python train_models.py")
        print("\nThis will train and save all ML models (takes 2-5 minutes)")
        return False
    
    print("✅ Models loaded successfully!")
    return True


@app.route('/')
def index():
    """Serve the main prediction interface"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        
        # Extract parameters
        weather = data.get('weather', 'Clear')
        hour = int(data.get('hour', 18))
        day_of_week = int(data.get('day_of_week', 0))
        distance = float(data.get('distance', 5.0))
        vehicle_type = data.get('vehicle_type', 'auto')
        month = int(data.get('month', 6))
        
        # Validate inputs
        if not (0 <= hour <= 23):
            return jsonify({'error': 'Hour must be between 0 and 23'}), 400
        if not (0 <= day_of_week <= 6):
            return jsonify({'error': 'Day of week must be between 0 (Mon) and 6 (Sun)'}), 400
        if distance <= 0:
            return jsonify({'error': 'Distance must be positive'}), 400
        
        # Calculate is_weekend
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Get predictions from all models
        results = {}
        
        # 1. Demand Forecasting
        try:
            demand_pred = ml_engine.predict_demand(
                hour=hour,
                day_of_week=day_of_week,
                month=month,
                is_weekend=is_weekend,
                weather_condition=weather
            )
            results['demand'] = {
                'value': float(demand_pred['Ensemble']),
                'level': 'High' if demand_pred['Ensemble'] > 150 else 'Medium' if demand_pred['Ensemble'] > 80 else 'Low',
                'all_predictions': convert_to_native(demand_pred)
            }
        except Exception as e:
            results['demand'] = {'error': str(e)}
        
        # 2. Dynamic Pricing (Main prediction)
        try:
            pricing_pred = ml_engine.predict_pricing(
                hour=hour,
                day_of_week=day_of_week,
                distance_km=distance,
                weather_condition=weather,
                vehicle_type=vehicle_type
            )
            
            # Calculate price breakdown
            surge_multiplier = float(pricing_pred['Ensemble'])
            base_fare = distance * 15  # ₹15 per km base rate
            
            # Weather premium calculation
            weather_premium_pct = 0
            if weather == 'Rain':
                weather_premium_pct = 38.3
            elif weather == 'Thunderstorm':
                weather_premium_pct = 45.0
            elif weather == 'Clouds':
                weather_premium_pct = 10.0
            
            weather_premium = base_fare * (weather_premium_pct / 100)
            
            # Surge pricing
            surge_amount = base_fare * (surge_multiplier - 1.0)
            
            # Total fare
            total_fare = base_fare + weather_premium + surge_amount
            
            results['pricing'] = {
                'base_fare': round(base_fare, 2),
                'weather_premium': round(weather_premium, 2),
                'weather_premium_pct': weather_premium_pct,
                'surge_multiplier': round(surge_multiplier, 2),
                'surge_amount': round(surge_amount, 2),
                'total_fare': round(total_fare, 2),
                'all_predictions': convert_to_native(pricing_pred)
            }
        except Exception as e:
            results['pricing'] = {'error': str(e)}
        
        # 3. Vehicle Recommendation
        try:
            vehicle_rec = ml_engine.recommend_vehicle(
                hour=hour,
                distance_km=distance,
                weather_condition=weather
            )
            results['vehicle_recommendation'] = {
                'recommended': vehicle_rec['RandomForest'],
                'confidence': float(max(vehicle_rec['Probabilities']['RandomForest'].values())),
                'all_options': convert_to_native(vehicle_rec['Probabilities']['RandomForest'])
            }
        except Exception as e:
            results['vehicle_recommendation'] = {'error': str(e)}
        
        # 4. Weather Impact
        try:
            weather_impact = ml_engine.predict_weather_impact(
                hour=hour,
                month=month,
                weather_condition=weather
            )
            results['weather_impact'] = {
                'score': round(float(weather_impact['Ensemble']), 2),
                'impact_level': 'High' if weather_impact['Ensemble'] > 1.3 else 'Medium' if weather_impact['Ensemble'] > 1.1 else 'Low',
                'all_predictions': convert_to_native(weather_impact)
            }
        except Exception as e:
            results['weather_impact'] = {'error': str(e)}
        
        return jsonify({
            'success': True,
            'predictions': results,
            'input_params': {
                'weather': weather,
                'hour': hour,
                'day_of_week': day_of_week,
                'distance': distance,
                'vehicle_type': vehicle_type,
                'month': month
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': ml_engine is not None
    })


def main():
    print("=" * 80)
    print("🚀 RAPIDO PRICE PREDICTION SERVER")
    print("=" * 80)
    
    # Load models
    if not load_models():
        return
    
    # Find available port
    port = find_free_port()
    
    print("\n" + "=" * 80)
    print("✅ SERVER READY")
    print("=" * 80)
    print(f"\n🌐 Open your browser and go to:")
    print(f"   http://localhost:{port}")
    print(f"\n💡 To stop the server: Press Ctrl+C")
    print("=" * 80 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == "__main__":
    main()

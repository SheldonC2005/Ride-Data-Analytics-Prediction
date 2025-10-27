"""
Quick Start Script - Rapido ML Price Predictor
Automatically trains models (if needed) and starts the web server
"""

import os
import sys
from pathlib import Path

def check_models_exist():
    """Check if trained models exist"""
    models_dir = Path("models")
    if not models_dir.exists():
        return False
    
    required_models = [
        'demand_rf.pkl', 'demand_lr.pkl',
        'pricing_xgb.pkl', 'pricing_gb.pkl'
    ]
    
    for model_file in required_models:
        if not (models_dir / model_file).exists():
            return False
    
    return True


def main():
    print("=" * 80)
    print("🚀 RAPIDO ML PRICE PREDICTOR - QUICK START")
    print("=" * 80)
    
    # Check if models exist
    if not check_models_exist():
        print("\n⚠️  No trained models found")
        print("\n📝 Starting one-time model training...")
        print("   This will take 2-5 minutes")
        print("-" * 80)
        
        # Run training script
        import train_models
        success = train_models.main()
        
        if not success:
            print("\n❌ Training failed. Please check the error messages above.")
            return False
        
        print("\n✅ Model training completed successfully!")
    else:
        print("\n✅ Trained models found - skipping training")
    
    # Start the prediction server
    print("\n" + "=" * 80)
    print("🌐 STARTING WEB SERVER")
    print("=" * 80)
    
    import prediction_server
    prediction_server.main()
    
    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)

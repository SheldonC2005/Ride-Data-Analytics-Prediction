"""
One-Time Model Training Script for Rapido ML Price Predictor
Run this script once to train all ML models and save them for fast predictions

Usage: python train_models.py
"""

import sys
import os
sys.path.append('scripts')

from ml_prediction_engine import RapidoMLEngine  # type: ignore
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 80)
    print("🚀 RAPIDO ML MODEL TRAINING")
    print("=" * 80)
    print("\nThis will train all 4 ML algorithm types:")
    print("  1. Demand Forecasting (RandomForest + Linear Regression)")
    print("  2. Dynamic Pricing (XGBoost + Gradient Boosting)")
    print("  3. Route Recommendation (Decision Tree + Random Forest)")
    print("  4. Weather Impact (Ridge Regression + XGBoost)")
    print("\nEstimated time: 2-5 minutes")
    print("-" * 80)

    # Initialize ML Engine
    print("\n📊 Step 1: Loading data...")
    engine = RapidoMLEngine("Dataset.csv")

    # Load and prepare data
    print("\n⚙️ Step 2: Preparing features...")
    engine.load_and_prepare_data()
    engine.prepare_ml_features()

    # Train all models
    print("\n" + "=" * 80)
    print("🎯 TRAINING ALL MODELS")
    print("=" * 80)

    try:
        # 1. Demand Forecasting
        print("\n[1/4] Training Demand Forecasting Models...")
        engine.train_demand_forecasting_models()

        # 2. Dynamic Pricing
        print("\n[2/4] Training Dynamic Pricing Models...")
        engine.train_pricing_models()

        # 3. Route Recommendation
        print("\n[3/4] Training Route Recommendation Models...")
        engine.train_route_recommendation_models()

        # 4. Weather Impact
        print("\n[4/4] Training Weather Impact Models...")
        engine.train_weather_impact_models()

        # Save all models
        print("\n" + "=" * 80)
        print("💾 SAVING TRAINED MODELS")
        print("=" * 80)
        engine.save_models()

        # Display summary
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE - MODEL SUMMARY")
        print("=" * 80)
        summary = engine.get_model_summary()

        print(f"\n📊 Total Models Trained: {summary['models_trained']}")
        print(f"🎯 Model Types: {', '.join(summary['model_types'])}")

        if 'best_models' in summary:
            print("\n🏆 Best Performing Models:")
            for model_type, performance in summary['best_models'].items():
                if 'R2' in performance:
                    print(f"  • {model_type}: {performance['model']} (R² = {performance['R2']:.3f})")
                elif 'Accuracy' in performance:
                    print(f"  • {model_type}: {performance['model']} (Accuracy = {performance['Accuracy']:.3f})")

        print("\n" + "=" * 80)
        print("🎉 SUCCESS! Models are ready for predictions")
        print("=" * 80)
        print("\n📝 Next Steps:")
        print("  1. Run: python prediction_server.py")
        print("  2. Open: http://localhost:5000 in your browser")
        print("  3. Enter ride details to get instant price predictions!")
        print("\n" + "=" * 80)

        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR DURING TRAINING")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        print("\nPlease check:")
        print("  • Dataset.csv exists in the project root")
        print("  • All required packages are installed (run: pip install -r requirements.txt)")
        print("  • Sufficient memory available")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

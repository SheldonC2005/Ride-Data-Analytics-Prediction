# 🚗 Rapido ML Price Predictor - User Guide

## 🎯 Quick Start (3 Steps)

### Step 1: Install Dependencies
```powershell
pip install Flask Flask-CORS
```

### Step 2: Train ML Models (One-time, ~2-5 minutes)
```powershell
python train_models.py
```

This will train all 4 ML algorithm types:
- ✅ Demand Forecasting (RandomForest + Linear Regression)
- ✅ Dynamic Pricing (XGBoost + Gradient Boosting)
- ✅ Route Recommendation (Decision Tree + Random Forest)
- ✅ Weather Impact (Ridge Regression + XGBoost)

### Step 3: Start the Web Server
```powershell
python prediction_server.py
```

Then open your browser to: **http://localhost:5000**

---

## 📝 How to Use

### Input Parameters:
1. **Weather Condition**: Clear, Clouds, Rain, Thunderstorm, Drizzle, or Mist
2. **Hour of Day**: 0-23 (24-hour format, e.g., 18 = 6 PM)
3. **Day of Week**: Monday to Sunday
4. **Distance**: Distance in kilometers
5. **Vehicle Type**: Auto, Bike, or Cab
6. **Month**: Current month (affects seasonal patterns)

### Click "Predict Price" to Get:

#### 💰 **Detailed Price Breakdown:**
- **Base Fare**: Distance × ₹15/km
- **Weather Premium**: Extra charge based on weather (Rain: +38.3%, Thunderstorm: +45%)
- **Surge Multiplier**: ML-predicted demand-based surge
- **Total Estimated Fare**: Complete price calculation

#### 📊 **Additional ML Predictions:**
- **Demand Forecast**: Expected number of trips per hour (Low/Medium/High)
- **Recommended Vehicle**: Best vehicle type for your conditions
- **Weather Impact Score**: How much weather affects pricing

---

## 🎨 Features

### All 4 ML Algorithm Types Integrated:
1. **Demand Forecasting** - Predicts ride demand
   - RandomForest: Handles non-linear patterns
   - Linear Regression: Captures linear trends
   - Ensemble: Combines both for accuracy

2. **Dynamic Pricing** - Calculates optimal surge multiplier
   - XGBoost: Gradient boosted decision trees
   - Gradient Boosting: Sequential error correction
   - Ensemble: Best of both algorithms

3. **Route Recommendation** - Suggests best vehicle
   - Decision Tree: Fast classification
   - Random Forest: Robust predictions
   - Probabilities: Confidence scores

4. **Weather Impact** - Quantifies weather effects
   - Ridge Regression: Regularized linear model
   - XGBoost: Complex pattern recognition
   - Ensemble: Balanced prediction

---

## 📊 Example Predictions

### Scenario 1: Rainy Evening Rush Hour
**Input:**
- Weather: Rain
- Hour: 18 (6 PM)
- Day: Monday
- Distance: 8 km
- Vehicle: Auto

**Predicted Output:**
- Base Fare: ₹120
- Weather Premium (+38.3%): ₹46
- Surge (2.1x): ₹132
- **Total: ₹298**
- Demand: High (180 trips/hour)

### Scenario 2: Clear Morning Commute
**Input:**
- Weather: Clear
- Hour: 8 (8 AM)
- Day: Wednesday
- Distance: 5 km
- Vehicle: Bike

**Predicted Output:**
- Base Fare: ₹75
- Weather Premium: ₹0
- Surge (1.3x): ₹22
- **Total: ₹97**
- Demand: Medium (120 trips/hour)

---

## 🔧 Troubleshooting

### "No trained models found"
**Solution:** Run `python train_models.py` first

### "Port already in use"
**Solution:** The server auto-finds available ports. Check console for actual port number.

### "Import error: flask"
**Solution:** Install Flask: `pip install Flask`

### Models not loading
**Solution:** Check if `models/` directory exists with `.pkl` files

---

## 📈 Model Performance Metrics

After training, you'll see accuracy metrics:

**Demand Forecasting:**
- R² Score: ~0.85-0.92
- MAE: ~8-12 trips

**Dynamic Pricing:**
- R² Score: ~0.78-0.88
- MAE: ~0.15-0.25 multiplier

**Route Recommendation:**
- Accuracy: ~82-89%

**Weather Impact:**
- R² Score: ~0.80-0.90
- MAE: ~0.10-0.18

---

## 🎯 Business Insights from ML

### Key Findings:
1. **Rain increases fares by 38.3% on average**
2. **Peak hours (17-19) have 1.5-2.2x surge**
3. **Weekend demand is 20% higher**
4. **Thunderstorms can add up to 45% premium**

### Optimal Conditions for Riders:
- Best time: Off-peak hours (10-15)
- Best weather: Clear conditions
- Best day: Tuesday-Wednesday
- Expected savings: 30-40% vs peak rain

### Optimal Conditions for Drivers:
- Best earnings: Evening rush (17-19) + Rain
- High demand: Weekends + VIT University routes
- Premium opportunities: Thunderstorm conditions

---

## 💡 Tips for Accurate Predictions

1. **Enter exact hour**: Use current hour for real-time accuracy
2. **Check weather**: Use actual current weather conditions
3. **Distance matters**: Longer distances amplify surge/weather effects
4. **Compare vehicles**: Try different vehicle types to find best value

---

## 🔐 Data Privacy

- All predictions run **locally on your machine**
- No data is sent to external servers
- ML models trained on historical data only
- Your ride details are never stored or transmitted

---

## 🚀 Advanced Usage

### API Endpoint (for developers):
```bash
POST http://localhost:5000/predict
Content-Type: application/json

{
  "weather": "Rain",
  "hour": 18,
  "day_of_week": 0,
  "distance": 8.0,
  "vehicle_type": "auto",
  "month": 6
}
```

### Health Check:
```bash
GET http://localhost:5000/health
```

---

## 📞 Support

If you encounter issues:
1. Check this README for troubleshooting
2. Verify all dependencies are installed
3. Ensure Dataset.csv is in project root
4. Check console output for error messages

---

**Powered by 4 ML Algorithm Types | 8 Trained Models | Real-time Predictions**

"""
Weather Forecast Integration for Rapido Transportation ML
Author: BI Project Team
Date: October 14, 2025

This module integrates weather forecasting capabilities using free APIs
for enhanced prediction accuracy in the ML models.
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class WeatherForecastAPI:
    """
    Weather forecast integration using OpenWeatherMap free API
    """
    
    def __init__(self, api_key=None):
        """
        Initialize weather API
        
        Args:
            api_key: OpenWeatherMap API key (get free at openweathermap.org)
        """
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.vellore_coords = {"lat": 12.9165, "lon": 79.1325}  # Vellore coordinates
        
        # Weather condition mappings
        self.weather_mapping = {
            'clear sky': 'Clear',
            'few clouds': 'Clouds',
            'scattered clouds': 'Clouds',
            'broken clouds': 'Clouds',
            'overcast clouds': 'Clouds',
            'shower rain': 'Rain',
            'rain': 'Rain',
            'light rain': 'Rain',
            'moderate rain': 'Rain',
            'heavy intensity rain': 'Rain',
            'thunderstorm': 'Thunderstorm',
            'thunderstorm with light rain': 'Thunderstorm',
            'thunderstorm with rain': 'Thunderstorm',
            'mist': 'Haze',
            'fog': 'Haze',
            'haze': 'Haze',
            'dust': 'Haze',
            'sand': 'Haze',
            'smoke': 'Haze',
            'drizzle': 'Rain',
            'light intensity drizzle': 'Rain',
            'snow': 'Clouds'  # Rare in Vellore, treat as clouds
        }
        
        print("🌤️ Weather Forecast API initialized")
    
    def get_current_weather(self):
        """Get current weather for Vellore"""
        if not self.api_key:
            return self._get_simulated_current_weather()
        
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': self.vellore_coords['lat'],
                'lon': self.vellore_coords['lon'],
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_weather_data(data)
            else:
                print(f"API Error: {response.status_code}")
                return self._get_simulated_current_weather()
                
        except Exception as e:
            print(f"Weather API error: {e}")
            return self._get_simulated_current_weather()
    
    def get_forecast(self, days=5):
        """
        Get weather forecast for next few days
        
        Args:
            days: Number of days to forecast (max 5 for free API)
        """
        if not self.api_key:
            return self._get_simulated_forecast(days)
        
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': self.vellore_coords['lat'],
                'lon': self.vellore_coords['lon'],
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_forecast_data(data, days)
            else:
                print(f"Forecast API Error: {response.status_code}")
                return self._get_simulated_forecast(days)
                
        except Exception as e:
            print(f"Forecast API error: {e}")
            return self._get_simulated_forecast(days)
    
    def _parse_weather_data(self, data):
        """Parse current weather API response"""
        weather_desc = data['weather'][0]['description'].lower()
        mapped_condition = self.weather_mapping.get(weather_desc, 'Clear')
        
        return {
            'datetime': datetime.now(),
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather_condition': mapped_condition,
            'weather_description': weather_desc,
            'wind_speed': data['wind']['speed'],
            'visibility': data.get('visibility', 10000) / 1000,  # Convert to km
            'rain_probability': self._calculate_rain_probability(data),
            'weather_impact_score': self._get_weather_impact_score(mapped_condition)
        }
    
    def _parse_forecast_data(self, data, days):
        """Parse forecast API response"""
        forecast_list = []
        
        # Process forecast data (3-hour intervals)
        for item in data['list'][:days*8]:  # 8 intervals per day
            weather_desc = item['weather'][0]['description'].lower()
            mapped_condition = self.weather_mapping.get(weather_desc, 'Clear')
            
            forecast_item = {
                'datetime': datetime.fromtimestamp(item['dt']),
                'temperature': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'pressure': item['main']['pressure'],
                'weather_condition': mapped_condition,
                'weather_description': weather_desc,
                'wind_speed': item['wind']['speed'],
                'rain_probability': self._calculate_rain_probability(item),
                'weather_impact_score': self._get_weather_impact_score(mapped_condition)
            }
            
            # Add rain volume if available
            if 'rain' in item:
                forecast_item['rain_volume'] = item['rain'].get('3h', 0)
            else:
                forecast_item['rain_volume'] = 0
            
            forecast_list.append(forecast_item)
        
        return pd.DataFrame(forecast_list)
    
    def _calculate_rain_probability(self, weather_data):
        """Calculate rain probability based on weather data"""
        # Check for rain in the data
        if 'rain' in weather_data:
            return min(1.0, weather_data['rain'].get('3h', 0) / 5.0)  # Scale by 5mm
        
        # Check humidity and weather condition
        humidity = weather_data['main']['humidity']
        weather_desc = weather_data['weather'][0]['description'].lower()
        
        if 'rain' in weather_desc or 'drizzle' in weather_desc:
            return 0.8
        elif 'thunderstorm' in weather_desc:
            return 0.9
        elif 'cloud' in weather_desc:
            return min(0.6, humidity / 100.0)
        else:
            return min(0.2, humidity / 200.0)
    
    def _get_weather_impact_score(self, weather_condition):
        """Get weather impact score for surge pricing"""
        impact_scores = {
            'Clear': 1.0,
            'Clouds': 1.1,
            'Haze': 1.3,
            'Rain': 1.9,
            'Thunderstorm': 2.3
        }
        return impact_scores.get(weather_condition, 1.0)
    
    def _get_simulated_current_weather(self):
        """Simulate current weather when API is unavailable"""
        # Use historical patterns from Vellore
        current_hour = datetime.now().hour
        current_month = datetime.now().month
        
        # Monsoon season (June-September) has higher rain probability
        if 6 <= current_month <= 9:
            rain_prob = 0.4
        else:
            rain_prob = 0.1
        
        # Afternoon hours have higher thunderstorm probability
        if 14 <= current_hour <= 17:
            rain_prob *= 1.5
        
        # Simulate weather based on probabilities
        rand = np.random.random()
        if rand < rain_prob * 0.3:
            weather_condition = 'Thunderstorm'
            temperature = np.random.uniform(24, 28)
            humidity = np.random.uniform(80, 95)
        elif rand < rain_prob:
            weather_condition = 'Rain'
            temperature = np.random.uniform(25, 30)
            humidity = np.random.uniform(70, 90)
        elif rand < rain_prob + 0.3:
            weather_condition = 'Clouds'
            temperature = np.random.uniform(26, 32)
            humidity = np.random.uniform(60, 80)
        else:
            weather_condition = 'Clear'
            temperature = np.random.uniform(28, 35)
            humidity = np.random.uniform(40, 70)
        
        return {
            'datetime': datetime.now(),
            'temperature': temperature,
            'humidity': humidity,
            'pressure': np.random.uniform(1010, 1020),
            'weather_condition': weather_condition,
            'weather_description': weather_condition.lower(),
            'wind_speed': np.random.uniform(2, 8),
            'visibility': np.random.uniform(8, 15),
            'rain_probability': rain_prob,
            'weather_impact_score': self._get_weather_impact_score(weather_condition)
        }
    
    def _get_simulated_forecast(self, days):
        """Simulate weather forecast when API is unavailable"""
        forecast_data = []
        
        for day in range(days):
            for hour in range(0, 24, 3):  # 3-hour intervals
                forecast_time = datetime.now() + timedelta(days=day, hours=hour)
                month = forecast_time.month
                hour_of_day = forecast_time.hour
                
                # Seasonal rain probability
                if 6 <= month <= 9:  # Monsoon
                    base_rain_prob = 0.4
                elif month in [10, 11]:  # Post-monsoon
                    base_rain_prob = 0.2
                else:  # Dry season
                    base_rain_prob = 0.1
                
                # Time-based adjustments
                if 14 <= hour_of_day <= 17:  # Afternoon thunderstorms
                    rain_prob = base_rain_prob * 1.5
                elif 6 <= hour_of_day <= 10:  # Morning clarity
                    rain_prob = base_rain_prob * 0.5
                else:
                    rain_prob = base_rain_prob
                
                # Generate weather
                rand = np.random.random()
                if rand < rain_prob * 0.3:
                    weather_condition = 'Thunderstorm'
                    temp = np.random.uniform(24, 28)
                    humidity = np.random.uniform(80, 95)
                    rain_volume = np.random.uniform(5, 20)
                elif rand < rain_prob:
                    weather_condition = 'Rain'
                    temp = np.random.uniform(25, 30)
                    humidity = np.random.uniform(70, 90)
                    rain_volume = np.random.uniform(1, 10)
                elif rand < rain_prob + 0.3:
                    weather_condition = 'Clouds'
                    temp = np.random.uniform(26, 32)
                    humidity = np.random.uniform(60, 80)
                    rain_volume = 0
                else:
                    weather_condition = 'Clear'
                    temp = np.random.uniform(28, 35)
                    humidity = np.random.uniform(40, 70)
                    rain_volume = 0
                
                forecast_data.append({
                    'datetime': forecast_time,
                    'temperature': temp,
                    'humidity': humidity,
                    'pressure': np.random.uniform(1010, 1020),
                    'weather_condition': weather_condition,
                    'weather_description': weather_condition.lower(),
                    'wind_speed': np.random.uniform(2, 8),
                    'rain_probability': rain_prob,
                    'rain_volume': rain_volume,
                    'weather_impact_score': self._get_weather_impact_score(weather_condition)
                })
        
        return pd.DataFrame(forecast_data)
    
    def get_hourly_forecast_features(self, hours_ahead=24):
        """
        Get weather features for next N hours for ML model input
        
        Args:
            hours_ahead: Number of hours to forecast
            
        Returns:
            DataFrame with hourly weather features
        """
        # Get forecast data
        forecast_df = self.get_forecast(days=min(5, hours_ahead // 24 + 1))
        
        if forecast_df.empty:
            return self._get_default_hourly_features(hours_ahead)
        
        # Resample to hourly data
        forecast_df.set_index('datetime', inplace=True)
        hourly_forecast = forecast_df.resample('H').mean()
        
        # Forward fill missing values
        hourly_forecast.ffill(inplace=True)
        
        # Add time-based features
        hourly_forecast['hour'] = hourly_forecast.index.hour
        hourly_forecast['day_of_week'] = hourly_forecast.index.dayofweek
        hourly_forecast['month'] = hourly_forecast.index.month
        hourly_forecast['is_weekend'] = hourly_forecast['day_of_week'].isin([5, 6]).astype(int)
        hourly_forecast['is_rush_hour'] = (
            ((hourly_forecast['hour'] >= 8) & (hourly_forecast['hour'] <= 9)) |
            ((hourly_forecast['hour'] >= 17) & (hourly_forecast['hour'] <= 18))
        ).astype(int)
        
        # Season encoding
        hourly_forecast['season'] = hourly_forecast['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'monsoon', 10: 'monsoon', 11: 'winter'
        })
        
        return hourly_forecast.head(hours_ahead)
    
    def _get_default_hourly_features(self, hours_ahead):
        """Generate default hourly features when forecast is unavailable"""
        current_time = datetime.now()
        default_data = []
        
        for i in range(hours_ahead):
            forecast_time = current_time + timedelta(hours=i)
            hour = forecast_time.hour
            month = forecast_time.month
            
            # Default weather based on time and season
            if 6 <= month <= 9:  # Monsoon
                weather_condition = 'Clouds' if hour % 6 == 0 else 'Rain'
            else:  # Dry season
                weather_condition = 'Clear' if hour % 8 != 0 else 'Clouds'
            
            default_data.append({
                'datetime': forecast_time,
                'temperature': 30.0,
                'humidity': 70.0,
                'pressure': 1015.0,
                'weather_condition': weather_condition,
                'wind_speed': 5.0,
                'rain_probability': 0.2,
                'rain_volume': 0.0,
                'weather_impact_score': self._get_weather_impact_score(weather_condition),
                'hour': hour,
                'day_of_week': forecast_time.weekday(),
                'month': month,
                'is_weekend': int(forecast_time.weekday() >= 5),
                'is_rush_hour': int((8 <= hour <= 9) or (17 <= hour <= 18)),
                'season': 'summer'
            })
        
        return pd.DataFrame(default_data).set_index('datetime')
    
    def save_forecast_data(self, filename="weather_forecast.xlsx"):
        """Save current forecast data to Excel for Power BI"""
        try:
            forecast_df = self.get_forecast(days=5)
            output_path = f"power_bi/data_model/{filename}"
            
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            forecast_df.to_excel(output_path, index=False, sheet_name="Weather_Forecast")
            print(f"✅ Weather forecast saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving forecast: {e}")
            return None
    
    def get_weather_alerts(self):
        """Get weather alerts for operational planning"""
        current_weather = self.get_current_weather()
        forecast_df = self.get_forecast(days=2)
        
        alerts = []
        
        # Current weather alerts
        if current_weather['weather_condition'] == 'Thunderstorm':
            alerts.append({
                'type': 'severe_weather',
                'severity': 'high',
                'message': 'Thunderstorm in progress - expect high surge pricing',
                'impact': 'Demand may decrease, but pricing will be premium',
                'recommendation': 'Increase vehicle availability in covered areas'
            })
        elif current_weather['weather_condition'] == 'Rain':
            alerts.append({
                'type': 'rain_alert',
                'severity': 'medium',
                'message': 'Rain detected - surge pricing likely',
                'impact': 'Increased ride demand and higher pricing',
                'recommendation': 'Deploy more vehicles to high-demand areas'
            })
        
        # Forecast alerts
        if not forecast_df.empty:
            # Check for upcoming severe weather
            severe_weather = forecast_df[
                forecast_df['weather_condition'].isin(['Thunderstorm', 'Rain'])
            ]
            
            if not severe_weather.empty:
                next_severe = severe_weather.iloc[0]
                alerts.append({
                    'type': 'forecast_alert',
                    'severity': 'medium',
                    'message': f"{next_severe['weather_condition']} expected at {next_severe['datetime'].strftime('%H:%M')}",
                    'impact': 'Plan for surge pricing and increased demand',
                    'recommendation': 'Pre-position vehicles in strategic locations'
                })
        
        return alerts

# Integration with ML Engine
class WeatherMLIntegration:
    """
    Integration class to connect weather forecasting with ML predictions
    """
    
    def __init__(self, ml_engine, weather_api):
        """
        Initialize integration
        
        Args:
            ml_engine: RapidoMLEngine instance
            weather_api: WeatherForecastAPI instance
        """
        self.ml_engine = ml_engine
        self.weather_api = weather_api
        
    def predict_with_weather_forecast(self, prediction_type, hours_ahead=24, **kwargs):
        """
        Make predictions using weather forecast data
        
        Args:
            prediction_type: 'demand', 'pricing', 'route', or 'weather_impact'
            hours_ahead: Number of hours to predict
            **kwargs: Additional parameters for predictions
        """
        # Get weather forecast
        weather_features = self.weather_api.get_hourly_forecast_features(hours_ahead)
        
        predictions = []
        
        for idx, weather_row in weather_features.iterrows():
            # Extract weather features
            weather_params = {
                'weather_condition': weather_row['weather_condition'],
                'weather_impact_score': weather_row['weather_impact_score'],
                'rain_probability': weather_row['rain_probability']
            }
            
            # Combine with other parameters
            params = {**kwargs, **weather_params}
            
            try:
                if prediction_type == 'demand':
                    pred = self.ml_engine.predict_demand(
                        hour=weather_row['hour'],
                        day_of_week=weather_row['day_of_week'],
                        month=weather_row['month'],
                        is_weekend=weather_row['is_weekend'],
                        **params
                    )
                elif prediction_type == 'pricing':
                    pred = self.ml_engine.predict_pricing(
                        hour=weather_row['hour'],
                        day_of_week=weather_row['day_of_week'],
                        **params
                    )
                elif prediction_type == 'route':
                    pred = self.ml_engine.recommend_vehicle(
                        hour=weather_row['hour'],
                        **params
                    )
                elif prediction_type == 'weather_impact':
                    pred = self.ml_engine.predict_weather_impact(
                        hour=weather_row['hour'],
                        month=weather_row['month'],
                        **params
                    )
                else:
                    raise ValueError(f"Unknown prediction type: {prediction_type}")
                
                predictions.append({
                    'datetime': idx,
                    'hour': weather_row['hour'],
                    'weather_condition': weather_row['weather_condition'],
                    'prediction': pred,
                    'weather_impact_score': weather_row['weather_impact_score']
                })
                
            except Exception as e:
                print(f"Prediction error for {idx}: {e}")
                continue
        
        return pd.DataFrame(predictions)
    
    def generate_operational_insights(self):
        """Generate operational insights combining weather and ML predictions"""
        current_weather = self.weather_api.get_current_weather()
        
        insights = {
            'current_conditions': current_weather,
            'alerts': self.weather_api.get_weather_alerts(),
            'recommendations': []
        }
        
        # Generate recommendations based on current weather
        if current_weather['weather_condition'] in ['Rain', 'Thunderstorm']:
            insights['recommendations'].extend([
                'Increase surge pricing by 20-50%',
                'Deploy additional vehicles to covered pickup points',
                'Alert drivers about weather conditions',
                'Monitor demand surge in real-time'
            ])
        elif current_weather['weather_condition'] == 'Clear':
            insights['recommendations'].extend([
                'Normal pricing strategy',
                'Focus on efficiency optimization',
                'Consider promotional pricing for off-peak hours'
            ])
        
        return insights

# Example usage
def main():
    """Test weather integration"""
    print("🌤️ Testing Weather Forecast Integration...")
    
    # Initialize weather API
    weather_api = WeatherForecastAPI()
    
    # Get current weather
    current_weather = weather_api.get_current_weather()
    print(f"Current weather: {current_weather}")
    
    # Get forecast
    forecast = weather_api.get_forecast(days=2)
    print(f"Forecast shape: {forecast.shape}")
    
    # Get hourly features
    hourly_features = weather_api.get_hourly_forecast_features(hours_ahead=12)
    print(f"Hourly features shape: {hourly_features.shape}")
    
    # Get alerts
    alerts = weather_api.get_weather_alerts()
    print(f"Weather alerts: {len(alerts)} alerts")
    
    # Save forecast data
    weather_api.save_forecast_data()
    
    print("✅ Weather integration test complete!")

if __name__ == "__main__":
    main()
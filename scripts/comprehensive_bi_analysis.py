"""
Comprehensive Business Intelligence Analysis for Rapido Transportation
Author: BI Project Team
Date: August 29, 2025

This script performs deep analysis on the transportation dataset covering:
1. All 45 KPIs and metrics identification
2. Time-based demand analysis (morning rush, evening rush, weekends, holidays)
3. Weather impact analysis with rain priority
4. Business insights and recommendations
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class RapidoBusinessIntelligence:
    """
    Comprehensive Business Intelligence Analysis for Rapido Transportation Data
    """

    def __init__(self, data_path):
        """Initialize with dataset path"""
        self.data_path = data_path
        self.df = None
        self.kpis = {}

    def load_and_prepare_data(self):
        """Load and prepare the dataset for analysis"""
        print("🚀 Loading Rapido Transportation Dataset...")
        self.df = pd.read_csv(self.data_path)

        # Convert datetime columns
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Create route column from from_location and to_location
        self.df['route'] = self.df['from_location'] + ' → ' + self.df['to_location']

        # Create additional time features for analysis
        self.df['month_name'] = self.df['datetime'].dt.month_name()
        self.df['day_name'] = self.df['datetime'].dt.day_name()
        self.df['year'] = self.df['datetime'].dt.year

        # Define rush hour periods
        self.df['is_morning_rush'] = (self.df['hour'] >= 8) & (self.df['hour'] <= 9)
        self.df['is_evening_rush'] = (self.df['hour'] >= 17) & (self.df['hour'] <= 18)
        self.df['is_rush_hour'] = self.df['is_morning_rush'] | self.df['is_evening_rush']

        # Define time periods with detailed categories
        def categorize_time_period(hour):
            if 0 <= hour < 6:
                return 'Late Night'
            elif 6 <= hour < 8:
                return 'Early Morning'
            elif 8 <= hour < 10:
                return 'Morning Rush'
            elif 10 <= hour < 12:
                return 'Late Morning'
            elif 12 <= hour < 14:
                return 'Lunch Time'
            elif 14 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 19:
                return 'Evening Rush'
            elif 19 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'

        self.df['detailed_time_period'] = self.df['hour'].apply(categorize_time_period)

        print(f"✅ Dataset loaded: {len(self.df):,} rides from {self.df['date'].min().date()} to {self.df['date'].max().date()}")
        return self.df

    def calculate_all_kpis(self):
        """Calculate all 45 KPIs and metrics"""
        print("\n📊 Calculating All KPIs and Metrics...")

        # 1-7: Revenue & Financial KPIs
        self.kpis['total_revenue'] = self.df['final_price'].sum()
        self.kpis['avg_revenue_per_ride'] = self.df['final_price'].mean()
        self.kpis['revenue_by_vehicle'] = self.df.groupby('vehicle_type')['final_price'].sum().to_dict()
        self.kpis['surge_revenue_impact'] = (self.df['final_price'] - self.df['base_price']).sum()
        self.kpis['avg_price_premium_percent'] = self.df['price_premium_percent'].mean()
        self.kpis['revenue_per_km'] = self.df['final_price'].sum() / self.df['distance_km'].sum()
        self.kpis['avg_cost_efficiency_ratio'] = (self.df['final_price'] / self.df['base_price']).mean()

        # 8-14: Operational KPIs
        self.kpis['total_trip_volume'] = len(self.df)
        self.kpis['avg_trip_distance'] = self.df['distance_km'].mean()
        self.kpis['fleet_utilization_by_vehicle'] = self.df['vehicle_type'].value_counts(normalize=True).to_dict()
        self.kpis['avg_waiting_time'] = self.df['waiting_time_minutes'].mean()
        self.kpis['trip_completion_rate'] = 1.0  # Assuming all trips in dataset are completed
        self.kpis['peak_hour_trip_distribution'] = self.df[self.df['is_rush_hour']]['hour'].value_counts().to_dict()
        self.kpis['distance_category_distribution'] = self.df['distance_category'].value_counts(normalize=True).to_dict()

        # 15-21: Time-Based Demand KPIs
        morning_rush = self.df[self.df['is_morning_rush']]
        evening_rush = self.df[self.df['is_evening_rush']]

        self.kpis['morning_rush_demand'] = len(morning_rush)
        self.kpis['evening_rush_demand'] = len(evening_rush)
        self.kpis['weekend_vs_weekday_performance'] = {
            'weekend': self.df[self.df['is_weekend']]['final_price'].sum(),
            'weekday': self.df[~self.df['is_weekend']]['final_price'].sum()
        }
        self.kpis['hourly_demand_pattern'] = self.df.groupby('hour').size().to_dict()
        self.kpis['day_of_week_performance'] = self.df.groupby('day_name')['final_price'].sum().to_dict()
        self.kpis['monthly_trends'] = self.df.groupby('month_name')['final_price'].sum().to_dict()

        rush_hour_premium = self.df[self.df['is_rush_hour']]['final_price'].mean()
        off_peak_premium = self.df[~self.df['is_rush_hour']]['final_price'].mean()
        self.kpis['rush_hour_premium'] = (rush_hour_premium - off_peak_premium) / off_peak_premium * 100

        # 22-28: Weather Impact KPIs
        rain_rides = self.df[self.df['weather_condition'] == 'Rain']
        clear_rides = self.df[self.df['weather_condition'] == 'Clear']

        self.kpis['weather_surge_frequency'] = self.df.groupby('weather_condition')['surge_multiplier'].mean().to_dict()
        self.kpis['rain_vs_clear_multiplier'] = rain_rides['surge_multiplier'].mean() / clear_rides['surge_multiplier'].mean() if len(clear_rides) > 0 else 0
        self.kpis['weather_demand_correlation'] = self.df.groupby('weather_condition').size().to_dict()
        self.kpis['precipitation_impact'] = self.df.groupby(pd.cut(self.df['precipitation'], bins=5))['surge_multiplier'].mean().to_dict()

        # Weather correlation with demand
        temp_corr = self.df['temperature'].corr(self.df['final_price'])
        self.kpis['temperature_demand_correlation'] = temp_corr

        visibility_impact = self.df.groupby(pd.cut(self.df['visibility'], bins=5)).size().to_dict()
        self.kpis['visibility_impact'] = visibility_impact

        storm_rides = self.df[self.df['weather_condition'] == 'Thunderstorm']
        self.kpis['storm_revenue_boost'] = storm_rides['surge_multiplier'].mean() if len(storm_rides) > 0 else 0

        # 29-34: Geographic & Route KPIs
        route_performance = self.df.groupby(['from_location', 'to_location']).agg({
            'final_price': ['sum', 'count'],
            'distance_km': 'mean'
        }).round(2)

        top_routes = route_performance['final_price']['sum'].nlargest(10)
        self.kpis['top_performing_routes'] = top_routes.to_dict()

        vit_routes = self.df[(self.df['from_location'].str.contains('VIT', na=False)
                              | self.df['to_location'].str.contains('VIT', na=False))]
        self.kpis['vit_university_performance'] = {
            'total_revenue': vit_routes['final_price'].sum(),
            'total_trips': len(vit_routes),
            'avg_price': vit_routes['final_price'].mean()
        }

        self.kpis['avg_distance_by_route'] = self.df.groupby(['from_location', 'to_location'])['distance_km'].mean().to_dict()

        # Route profitability (revenue per km)
        route_profitability = self.df.groupby(['from_location', 'to_location']).apply(
            lambda x: x['final_price'].sum() / x['distance_km'].sum()
        ).nlargest(10)
        self.kpis['route_profitability_index'] = route_profitability.to_dict()

        self.kpis['popular_destinations'] = self.df['to_location'].value_counts().head(10).to_dict()

        # 35-40: Business Intelligence KPIs
        demand_score = self.df.groupby(['from_location', 'to_location']).size()
        self.kpis['customer_demand_score'] = demand_score.describe().to_dict()

        # Surge pricing effectiveness (conversion rate)
        surge_effectiveness = self.df[self.df['surge_multiplier'] > 1.2]['final_price'].sum() / self.df['final_price'].sum()
        self.kpis['surge_pricing_effectiveness'] = surge_effectiveness

        # Market penetration by area
        area_penetration = self.df.groupby('from_location').size().to_dict()
        self.kpis['market_penetration_by_area'] = area_penetration

        # Service quality index (inverse of waiting time)
        avg_wait_by_area = self.df.groupby('from_location')['waiting_time_minutes'].mean()
        self.kpis['service_quality_index'] = (1 / avg_wait_by_area).to_dict()

        # Price competitiveness (compared to base price)
        price_competitiveness = (self.df['final_price'] / self.df['base_price']).mean()
        self.kpis['competitive_pricing_analysis'] = price_competitiveness

        # Growth rate calculation (month over month)
        monthly_revenue = self.df.groupby(['year', 'month'])['final_price'].sum()
        if len(monthly_revenue) > 1:
            growth_rate = ((monthly_revenue.iloc[-1] - monthly_revenue.iloc[0]) / monthly_revenue.iloc[0]) * 100
            self.kpis['revenue_growth_rate'] = growth_rate
        else:
            self.kpis['revenue_growth_rate'] = 0

        # 41-45: Strategic KPIs
        vehicle_revenue_share = self.df.groupby('vehicle_type')['final_price'].sum()
        total_revenue = vehicle_revenue_share.sum()
        self.kpis['revenue_diversification'] = (vehicle_revenue_share / total_revenue * 100).to_dict()

        # Demand forecasting accuracy (using standard deviation as proxy)
        demand_std = self.df.groupby('hour').size().std()
        self.kpis['demand_forecasting_accuracy'] = 100 - (demand_std / self.df.groupby('hour').size().mean() * 100)

        # Price elasticity (correlation between price and demand)
        price_elasticity = self.df['final_price'].corr(self.df.groupby('hour').transform('size'))
        self.kpis['price_elasticity_of_demand'] = price_elasticity

        # Customer satisfaction proxy
        satisfaction_proxy = (
            (self.df['waiting_time_minutes'] <= 5).mean() * 0.6
            + (self.df['price_premium_percent'] <= 30).mean() * 0.4
        ) * 100
        self.kpis['customer_satisfaction_proxy'] = satisfaction_proxy

        # Business sustainability index
        sustainability_factors = {
            'revenue_stability': 1 - self.df.groupby('date')['final_price'].sum().std() / self.df.groupby('date')['final_price'].sum().mean(),
            'demand_consistency': 1 - self.df.groupby('hour').size().std() / self.df.groupby('hour').size().mean(),
            'weather_resilience': len(self.df[self.df['weather_condition'] != 'Clear']) / len(self.df)
        }
        self.kpis['business_sustainability_index'] = np.mean(list(sustainability_factors.values())) * 100

        print("✅ All 45 KPIs calculated successfully!")
        return self.kpis

    def create_python_visualizations(self):
        """Generate comprehensive Python visualizations for analysis"""
        print("\n🎨 Creating Python Visualizations...")

        # Create visualizations directory
        viz_dir = "visualizations/charts"
        os.makedirs(viz_dir, exist_ok=True)

        # 1. Revenue Trend Analysis
        self._create_revenue_trend_chart(viz_dir)

        # 2. Time-based Analysis
        self._create_time_analysis_charts(viz_dir)

        # 3. Weather Impact Analysis
        self._create_weather_analysis_charts(viz_dir)

        # 4. Route Performance Analysis
        self._create_route_analysis_charts(viz_dir)

        # 5. Vehicle Analysis
        self._create_vehicle_analysis_charts(viz_dir)

        # 6. Interactive Dashboard
        self._create_interactive_dashboard(viz_dir)

        print(f"✅ All visualizations saved to: {viz_dir}")
        return viz_dir

    def _create_revenue_trend_chart(self, viz_dir):
        """Create revenue trend analysis charts"""
        # Monthly revenue trend
        monthly_revenue = self.df.groupby(self.df['date'].dt.to_period('M'))['final_price'].sum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_revenue.index.astype(str),
            y=monthly_revenue.values,
            mode='lines+markers',
            name='Monthly Revenue',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8)
        ))

        fig.update_layout(
            title='Monthly Revenue Trend - Rapido Transportation',
            xaxis_title='Month',
            yaxis_title='Revenue (₹)',
            template='plotly_white',
            height=400
        )

        fig.write_html(f"{viz_dir}/revenue_trend.html")
        print("  ✓ Revenue trend chart created")

    def _create_time_analysis_charts(self, viz_dir):
        """Create time-based analysis charts"""
        # Hourly demand pattern
        hourly_data = self.df.groupby('hour').agg({
            'final_price': ['sum', 'count', 'mean'],
            'surge_multiplier': 'mean'
        }).round(2)

        hourly_data.columns = ['Total_Revenue', 'Trip_Count', 'Avg_Revenue', 'Avg_Multiplier']

        # Create subplot for hourly analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Hourly Trip Volume', 'Hourly Revenue', 'Average Multiplier by Hour', 'Rush Hour vs Regular'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Trip volume by hour
        fig.add_trace(
            go.Bar(x=hourly_data.index, y=hourly_data['Trip_Count'], name='Trip Count'),
            row=1, col=1
        )

        # Revenue by hour
        fig.add_trace(
            go.Scatter(x=hourly_data.index, y=hourly_data['Total_Revenue'],
                       mode='lines+markers', name='Revenue'),
            row=1, col=2
        )

        # Multiplier by hour
        fig.add_trace(
            go.Bar(x=hourly_data.index, y=hourly_data['Avg_Multiplier'], name='Avg Multiplier'),
            row=2, col=1
        )

        # Rush hour comparison
        rush_comparison = self.df.groupby('is_rush_hour')['final_price'].agg(['sum', 'count'])
        fig.add_trace(
            go.Bar(x=['Regular Hours', 'Rush Hours'], y=rush_comparison['sum'], name='Revenue Comparison'),
            row=2, col=2
        )

        fig.update_layout(height=800, title_text="Time-Based Analysis Dashboard")
        fig.write_html(f"{viz_dir}/time_analysis.html")
        print("  ✓ Time analysis charts created")

    def _create_weather_analysis_charts(self, viz_dir):
        """Create weather impact analysis charts"""
        weather_analysis = self.df.groupby('weather_condition').agg({
            'final_price': ['sum', 'mean', 'count'],
            'surge_multiplier': 'mean',
            'waiting_time_minutes': 'mean'
        }).round(2)

        weather_analysis.columns = ['Total_Revenue', 'Avg_Revenue', 'Trip_Count', 'Avg_Multiplier', 'Avg_Wait']

        # Weather impact visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue by Weather', 'Average Multiplier', 'Trip Count', 'Waiting Time'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

        fig.add_trace(
            go.Bar(x=weather_analysis.index, y=weather_analysis['Total_Revenue'],
                   marker_color=colors, name='Revenue'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=weather_analysis.index, y=weather_analysis['Avg_Multiplier'],
                   marker_color=colors, name='Multiplier'),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(x=weather_analysis.index, y=weather_analysis['Trip_Count'],
                   marker_color=colors, name='Trips'),
            row=2, col=1
        )

        fig.add_trace(
            go.Bar(x=weather_analysis.index, y=weather_analysis['Avg_Wait'],
                   marker_color=colors, name='Wait Time'),
            row=2, col=2
        )

        fig.update_layout(height=600, title_text="Weather Impact Analysis")
        fig.write_html(f"{viz_dir}/weather_analysis.html")
        print("  ✓ Weather analysis charts created")

    def _create_route_analysis_charts(self, viz_dir):
        """Create route performance analysis"""
        # Top 10 routes by revenue
        top_routes = self.df.groupby('route')['final_price'].sum().sort_values(ascending=False).head(10)

        fig = go.Figure(data=[
            go.Bar(x=top_routes.values, y=top_routes.index, orientation='h',
                   marker_color='#2E86AB')
        ])

        fig.update_layout(
            title='Top 10 Routes by Revenue',
            xaxis_title='Revenue (₹)',
            yaxis_title='Route',
            height=500
        )

        fig.write_html(f"{viz_dir}/route_analysis.html")
        print("  ✓ Route analysis chart created")

    def _create_vehicle_analysis_charts(self, viz_dir):
        """Create vehicle performance analysis"""
        vehicle_data = self.df.groupby('vehicle_type').agg({
            'final_price': ['sum', 'mean', 'count'],
            'distance_km': 'mean',
            'waiting_time_minutes': 'mean'
        }).round(2)

        vehicle_data.columns = ['Total_Revenue', 'Avg_Revenue', 'Trip_Count', 'Avg_Distance', 'Avg_Wait']

        # Vehicle performance pie chart
        fig = go.Figure(data=[go.Pie(
            labels=vehicle_data.index,
            values=vehicle_data['Total_Revenue'],
            hole=0.4,
            marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )])

        fig.update_layout(
            title='Revenue Distribution by Vehicle Type',
            height=400
        )

        fig.write_html(f"{viz_dir}/vehicle_analysis.html")
        print("  ✓ Vehicle analysis chart created")

    def _create_interactive_dashboard(self, viz_dir):
        """Create comprehensive interactive dashboard"""
        # Create a comprehensive dashboard combining all insights
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Monthly Revenue', 'Hourly Demand', 'Weather Impact',
                            'Vehicle Distribution', 'Top Routes', 'Rush Hour Analysis',
                            'Premium Distribution', 'VIT vs Non-VIT', 'Service Quality'),
            specs=[[{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "bar"}, {"type": "indicator"}]]
        )

        # Monthly revenue trend
        monthly_revenue = self.df.groupby(self.df['date'].dt.to_period('M'))['final_price'].sum()
        fig.add_trace(
            go.Scatter(x=monthly_revenue.index.astype(str), y=monthly_revenue.values,
                       mode='lines+markers', name='Monthly Revenue'),
            row=1, col=1
        )

        # Hourly demand
        hourly_trips = self.df.groupby('hour').size()
        fig.add_trace(
            go.Bar(x=hourly_trips.index, y=hourly_trips.values, name='Hourly Trips'),
            row=1, col=2
        )

        # Weather impact
        weather_revenue = self.df.groupby('weather_condition')['final_price'].sum()
        fig.add_trace(
            go.Bar(x=weather_revenue.index, y=weather_revenue.values, name='Weather Revenue'),
            row=1, col=3
        )

        # Vehicle distribution
        vehicle_revenue = self.df.groupby('vehicle_type')['final_price'].sum()
        fig.add_trace(
            go.Pie(labels=vehicle_revenue.index, values=vehicle_revenue.values, name='Vehicle Revenue'),
            row=2, col=1
        )

        # Top 5 routes
        top_routes = self.df.groupby('route')['final_price'].sum().sort_values(ascending=False).head(5)
        fig.add_trace(
            go.Bar(x=top_routes.index, y=top_routes.values, name='Top Routes'),
            row=2, col=2
        )

        # Rush hour analysis
        rush_data = self.df.groupby('is_rush_hour')['final_price'].sum()
        fig.add_trace(
            go.Bar(x=['Regular', 'Rush'], y=rush_data.values, name='Rush Analysis'),
            row=2, col=3
        )

        # Premium distribution
        fig.add_trace(
            go.Histogram(x=self.df['price_premium_percent'], name='Premium Distribution'),
            row=3, col=1
        )

        # VIT analysis
        vit_revenue = [
            self.df[self.df['route'].str.contains('VIT', na=False)]['final_price'].sum(),
            self.df[~self.df['route'].str.contains('VIT', na=False)]['final_price'].sum()
        ]
        fig.add_trace(
            go.Bar(x=['VIT Related', 'Non-VIT'], y=vit_revenue, name='VIT Analysis'),
            row=3, col=2
        )

        # Service quality indicator
        avg_wait = self.df['waiting_time_minutes'].mean()
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=avg_wait,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Avg Wait (min)"},
                gauge={'axis': {'range': [None, 10]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 5], 'color': "lightgray"},
                                 {'range': [5, 10], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                     'thickness': 0.75, 'value': 7}}
            ),
            row=3, col=3
        )

        fig.update_layout(
            height=1200,
            title_text="Rapido Transportation - Comprehensive Analytics Dashboard",
            showlegend=False
        )

        fig.write_html(f"{viz_dir}/comprehensive_dashboard.html")
        print("  ✓ Comprehensive interactive dashboard created")

    def generate_power_bi_insights(self):
        """Generate insights specifically for Power BI dashboard creation"""
        print("\n🔄 Generating Power BI Integration Insights...")

        insights = {
            'key_metrics': {
                'total_revenue': f"₹{self.kpis['total_revenue']:,.2f}",
                'total_trips': f"{self.kpis['total_trip_volume']:,}",
                'avg_revenue_per_ride': f"₹{self.kpis['avg_revenue_per_ride']:.2f}",
                'vit_dependency': f"{(self.kpis['vit_university_performance']['total_revenue'] / self.kpis['total_revenue'] * 100):.1f}%"
            },
            'time_patterns': {
                'peak_hour': self.df.groupby('hour')['final_price'].sum().idxmax(),
                'rush_hour_premium': f"{((self.df[self.df['is_rush_hour']]['final_price'].mean() / self.df[~self.df['is_rush_hour']]['final_price'].mean()) - 1) * 100:.1f}%"
            },
            'weather_insights': {
                'rain_premium': f"{((self.df[self.df['weather_condition'] == 'Rain']['final_price'].mean() / self.df[self.df['weather_condition'] == 'Clear']['final_price'].mean()) - 1) * 100:.1f}%",
                'best_weather_for_revenue': self.df.groupby('weather_condition')['final_price'].sum().idxmax()
            },
            'route_insights': {
                'top_route': self.df.groupby('route')['final_price'].sum().idxmax(),
                'most_frequent_route': self.df['route'].mode().iloc[0]
            }
        }

        return insights

    def analyze_rush_hour_patterns(self):
        """Detailed analysis of morning and evening rush hours"""
        print("\n⏰ Analyzing Rush Hour Patterns...")

        # Morning rush hour analysis (8-9 AM)
        morning_rush = self.df[self.df['is_morning_rush']]

        # Evening rush hour analysis (5:30-6:30 PM)
        evening_rush = self.df[self.df['is_evening_rush']]

        rush_hour_analysis = {
            'morning_rush': {
                'total_trips': len(morning_rush),
                'total_revenue': morning_rush['final_price'].sum(),
                'avg_surge_multiplier': morning_rush['surge_multiplier'].mean(),
                'avg_waiting_time': morning_rush['waiting_time_minutes'].mean(),
                'most_popular_routes': morning_rush.groupby(['from_location', 'to_location']).size().nlargest(5).to_dict(),
                'vehicle_preference': morning_rush['vehicle_type'].value_counts().to_dict(),
                'avg_distance': morning_rush['distance_km'].mean(),
                'weather_impact': morning_rush.groupby('weather_condition')['surge_multiplier'].mean().to_dict()
            },
            'evening_rush': {
                'total_trips': len(evening_rush),
                'total_revenue': evening_rush['final_price'].sum(),
                'avg_surge_multiplier': evening_rush['surge_multiplier'].mean(),
                'avg_waiting_time': evening_rush['waiting_time_minutes'].mean(),
                'most_popular_routes': evening_rush.groupby(['from_location', 'to_location']).size().nlargest(5).to_dict(),
                'vehicle_preference': evening_rush['vehicle_type'].value_counts().to_dict(),
                'avg_distance': evening_rush['distance_km'].mean(),
                'weather_impact': evening_rush.groupby('weather_condition')['surge_multiplier'].mean().to_dict()
            }
        }

        # Weekend vs Weekday analysis
        weekend_data = self.df[self.df['is_weekend']]
        weekday_data = self.df[~self.df['is_weekend']]

        weekend_analysis = {
            'weekend_patterns': {
                'peak_hours': weekend_data.groupby('hour').size().nlargest(3).index.tolist(),
                'total_revenue': weekend_data['final_price'].sum(),
                'avg_trip_distance': weekend_data['distance_km'].mean(),
                'popular_destinations': weekend_data['to_location'].value_counts().head(5).to_dict(),
                'weather_sensitivity': weekend_data.groupby('weather_condition')['surge_multiplier'].mean().to_dict()
            },
            'weekday_patterns': {
                'peak_hours': weekday_data.groupby('hour').size().nlargest(3).index.tolist(),
                'total_revenue': weekday_data['final_price'].sum(),
                'avg_trip_distance': weekday_data['distance_km'].mean(),
                'popular_destinations': weekday_data['to_location'].value_counts().head(5).to_dict(),
                'weather_sensitivity': weekday_data.groupby('weather_condition')['surge_multiplier'].mean().to_dict()
            }
        }

        return rush_hour_analysis, weekend_analysis

    def analyze_weather_impact(self):
        """Comprehensive weather impact analysis with rain priority"""
        print("\n🌦️ Analyzing Weather Impact with Rain Priority...")

        weather_analysis = {}

        # Overall weather impact
        weather_stats = self.df.groupby('weather_condition').agg({
            'final_price': ['mean', 'sum', 'count'],
            'surge_multiplier': ['mean', 'max'],
            'waiting_time_minutes': 'mean',
            'distance_km': 'mean',
            'price_premium_percent': 'mean'
        }).round(2)

        weather_analysis['overall_impact'] = weather_stats.to_dict()

        # Rain-specific analysis (priority focus)
        rain_data = self.df[self.df['weather_condition'] == 'Rain']
        clear_data = self.df[self.df['weather_condition'] == 'Clear']

        rain_analysis = {
            'rain_premium': {
                'avg_surge_multiplier': rain_data['surge_multiplier'].mean(),
                'max_surge_seen': rain_data['surge_multiplier'].max(),
                'revenue_boost': (rain_data['final_price'].sum() - rain_data['base_price'].sum()),
                'trip_volume': len(rain_data),
                'avg_waiting_time': rain_data['waiting_time_minutes'].mean(),
                'popular_vehicle_in_rain': rain_data['vehicle_type'].mode().iloc[0] if len(rain_data) > 0 else 'N/A'
            },
            'rain_vs_clear_comparison': {
                'surge_difference': rain_data['surge_multiplier'].mean() - clear_data['surge_multiplier'].mean() if len(clear_data) > 0 else 0,
                'price_difference': rain_data['final_price'].mean() - clear_data['final_price'].mean() if len(clear_data) > 0 else 0,
                'demand_difference': len(rain_data) - len(clear_data),
                'waiting_time_difference': rain_data['waiting_time_minutes'].mean() - clear_data['waiting_time_minutes'].mean() if len(clear_data) > 0 else 0
            }
        }

        weather_analysis['rain_focus'] = rain_analysis

        # Seasonal weather patterns
        seasonal_weather = self.df.groupby(['month_name', 'weather_condition']).size().unstack(fill_value=0)
        weather_analysis['seasonal_patterns'] = seasonal_weather.to_dict()

        # Weather impact by time of day
        hourly_weather_impact = self.df.groupby(['hour', 'weather_condition'])['surge_multiplier'].mean().unstack(fill_value=0)
        weather_analysis['hourly_weather_impact'] = hourly_weather_impact.to_dict()

        # Extreme weather events
        extreme_weather = self.df[self.df['weather_condition'].isin(['Thunderstorm', 'Rain'])]
        extreme_analysis = {
            'extreme_weather_revenue': extreme_weather['final_price'].sum(),
            'extreme_weather_trips': len(extreme_weather),
            'avg_extreme_surge': extreme_weather['surge_multiplier'].mean(),
            'extreme_weather_share': len(extreme_weather) / len(self.df) * 100
        }

        weather_analysis['extreme_weather'] = extreme_analysis

        return weather_analysis

    def generate_business_insights(self):
        """Generate actionable business insights and recommendations"""
        print("\n💡 Generating Business Insights and Recommendations...")

        insights = {
            'revenue_insights': [],
            'operational_insights': [],
            'strategic_recommendations': [],
            'weather_strategy': [],
            'time_optimization': []
        }

        # Revenue insights
        total_revenue = self.kpis['total_revenue']
        surge_revenue = self.kpis['surge_revenue_impact']
        surge_contribution = (surge_revenue / total_revenue) * 100

        insights['revenue_insights'].append(f"💰 Total Revenue: ₹{total_revenue:,.2f}")
        insights['revenue_insights'].append(f"📈 Surge Pricing contributes {surge_contribution:.1f}% of total revenue")
        insights['revenue_insights'].append(f"🚗 Vehicle Revenue Distribution: {self.kpis['revenue_diversification']}")

        # Peak hour insights
        if self.kpis['evening_rush_demand'] > self.kpis['morning_rush_demand']:
            insights['time_optimization'].append("🌆 Evening rush (5:30-6:30 PM) shows higher demand than morning rush")
        else:
            insights['time_optimization'].append("🌅 Morning rush (8-9 AM) shows higher demand than evening rush")

        insights['time_optimization'].append(f"⚡ Rush hour premium: {self.kpis['rush_hour_premium']:.1f}% higher pricing")

        # Weather insights
        rain_multiplier = self.kpis.get('rain_vs_clear_multiplier', 0)
        if rain_multiplier > 1.5:
            insights['weather_strategy'].append(f"☔ Rain creates {rain_multiplier:.1f}x higher surge pricing opportunity")

        # Operational insights
        avg_wait = self.kpis['avg_waiting_time']
        if avg_wait > 5:
            insights['operational_insights'].append(f"⏱️ Average waiting time ({avg_wait:.1f} min) needs improvement")
        else:
            insights['operational_insights'].append(f"✅ Good service quality with {avg_wait:.1f} min average wait time")

        # VIT University insights
        vit_performance = self.kpis['vit_university_performance']
        vit_revenue_share = (vit_performance['total_revenue'] / total_revenue) * 100
        insights['strategic_recommendations'].append(f"🎓 VIT University routes contribute {vit_revenue_share:.1f}% of total revenue")

        # Top recommendations based on data
        if surge_contribution > 30:
            insights['strategic_recommendations'].append("🎯 Focus on surge pricing optimization - it's a major revenue driver")

        if self.kpis['customer_satisfaction_proxy'] < 70:
            insights['strategic_recommendations'].append("👥 Customer satisfaction needs attention - optimize wait times and pricing")

        return insights

    def create_summary_report(self):
        """Create a comprehensive summary report"""
        print("\n📋 Creating Summary Report...")

        report = {
            'executive_summary': {
                'total_rides': f"{len(self.df):,}",
                'total_revenue': f"₹{self.kpis['total_revenue']:,.2f}",
                'avg_revenue_per_ride': f"₹{self.kpis['avg_revenue_per_ride']:.2f}",
                'data_period': f"{self.df['date'].min().date()} to {self.df['date'].max().date()}",
                'surge_contribution': f"{(self.kpis['surge_revenue_impact'] / self.kpis['total_revenue']) * 100:.1f}%"
            },
            'key_performance_indicators': self.kpis,
            'business_insights': self.generate_business_insights()
        }

        return report

    def save_results(self, output_dir="../analysis/business_insights/"):
        """Save all analysis results to files"""
        import os
        import json

        os.makedirs(output_dir, exist_ok=True)

        # Clean KPIs for JSON serialization
        cleaned_kpis = {}
        for key, value in self.kpis.items():
            if isinstance(value, dict):
                cleaned_value = {}
                for k, v in value.items():
                    # Convert complex keys to strings
                    str_key = str(k) if not isinstance(k, (str, int, float, bool)) else k
                    cleaned_value[str_key] = str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                cleaned_kpis[key] = cleaned_value
            else:
                cleaned_kpis[key] = str(value) if not isinstance(value, (str, int, float, bool, type(None))) else value

        # Save KPIs
        with open(f"{output_dir}all_kpis.json", 'w') as f:
            json.dump(cleaned_kpis, f, indent=2, default=str)

        # Helper function to clean data for JSON
        def clean_for_json(obj):
            import numpy as np
            if isinstance(obj, dict):
                cleaned = {}
                for k, v in obj.items():
                    clean_key = str(k)
                    cleaned[clean_key] = clean_for_json(v)
                return cleaned
            elif isinstance(obj, (list, tuple)):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)

        # Save rush hour analysis
        rush_hour_analysis, weekend_analysis = self.analyze_rush_hour_patterns()
        with open(f"{output_dir}rush_hour_analysis.json", 'w') as f:
            json.dump(clean_for_json(rush_hour_analysis), f, indent=2, default=str)

        with open(f"{output_dir}weekend_analysis.json", 'w') as f:
            json.dump(clean_for_json(weekend_analysis), f, indent=2, default=str)

        # Save weather analysis
        weather_analysis = self.analyze_weather_impact()
        with open(f"{output_dir}weather_analysis.json", 'w') as f:
            json.dump(clean_for_json(weather_analysis), f, indent=2, default=str)

        # Save summary report
        summary_report = self.create_summary_report()
        with open(f"{output_dir}summary_report.json", 'w') as f:
            json.dump(clean_for_json(summary_report), f, indent=2, default=str)

        print(f"✅ All analysis results saved to {output_dir}")
        return output_dir


def main():
    """Main execution function with hybrid Python + Power BI approach"""
    print("🚀 Starting Comprehensive BI Analysis for Rapido Transportation")
    print("=" * 70)

    # Initialize analyzer
    analyzer = RapidoBusinessIntelligence("Dataset.csv")

    # Load and prepare data
    analyzer.load_and_prepare_data()

    # Calculate all KPIs
    analyzer.calculate_all_kpis()

    # Generate Python visualizations
    print("\n🎨 Creating Interactive Python Visualizations...")
    viz_dir = analyzer.create_python_visualizations()

    # Generate Power BI insights
    power_bi_insights = analyzer.generate_power_bi_insights()

    # Perform specialized analyses
    rush_hour_analysis, weekend_analysis = analyzer.analyze_rush_hour_patterns()
    analyzer.analyze_weather_impact()

    # Generate insights
    insights = analyzer.generate_business_insights()

    # Create and display summary
    summary = analyzer.create_summary_report()

    print("\n" + "=" * 70)
    print("📊 EXECUTIVE SUMMARY")
    print("=" * 70)

    for key, value in summary['executive_summary'].items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    print("\n💡 KEY INSIGHTS:")
    for category, insight_list in insights.items():
        if insight_list:
            print(f"\n{category.replace('_', ' ').title()}:")
            for insight in insight_list[:3]:  # Show top 3 insights per category
                print(f"  • {insight}")

    print("\n🔗 POWER BI INTEGRATION READY:")
    print(f"📈 Key Metrics: {power_bi_insights['key_metrics']}")
    print(f"⏰ Time Patterns: {power_bi_insights['time_patterns']}")
    print(f"🌦️ Weather Insights: {power_bi_insights['weather_insights']}")
    print(f"🛣️ Route Insights: {power_bi_insights['route_insights']}")

    # Save results including Power BI integration data
    output_dir = analyzer.save_results()

    # Save Power BI insights
    import json
    with open(f"{output_dir}power_bi_insights.json", 'w') as f:
        json.dump(power_bi_insights, f, indent=2, default=str)

    print("\n✅ Hybrid Analysis Complete!")
    print(f"📁 Python visualizations: {viz_dir}")
    print(f"📁 Analysis results: {output_dir}")
    print("🔄 Ready for Power BI dashboard creation...")

    return analyzer, summary, power_bi_insights


class RapidoMLBIIntegration:
    """
    Integration class that combines ML predictions with BI analysis
    """

    def __init__(self, data_path):
        """Initialize ML-BI integration"""
        self.data_path = data_path
        self.bi_analyzer = RapidoBusinessIntelligence(data_path)
        self.ml_engine = None
        self.weather_api = None
        self.geo_engine = None

    def setup_ml_components(self):
        """Setup ML components"""
        print("🤖 Setting up ML components...")

        # Import ML modules
        try:
            from ml_prediction_engine import RapidoMLEngine
            from weather_forecast_integration import WeatherForecastAPI
            from geospatial_features import GeospatialFeatureEngine

            # Initialize components
            self.ml_engine = RapidoMLEngine(self.data_path)
            self.weather_api = WeatherForecastAPI()
            self.geo_engine = GeospatialFeatureEngine()

            print("✅ ML components initialized")
            return True
        except ImportError as e:
            print(f"⚠️ ML components not available: {e}")
            return False

    def run_complete_analysis(self):
        """Run complete analysis with ML and BI integration"""
        print("🚀 Starting Complete ML + BI Analysis...")

        # 1. Traditional BI Analysis
        print("\n📊 Phase 1: Business Intelligence Analysis")
        df = self.bi_analyzer.load_and_prepare_data()
        self.bi_analyzer.calculate_all_kpis()
        self.bi_analyzer.create_python_visualizations()

        # 2. ML Enhancement (if available)
        ml_available = self.setup_ml_components()

        if ml_available:
            print("\n🤖 Phase 2: Machine Learning Enhancement")

            # Enhance data with geospatial features
            enhanced_df = self.geo_engine.process_dataset(df)

            # Setup ML engine with enhanced data
            self.ml_engine.df = enhanced_df
            self.ml_engine.prepare_ml_features()

            # Train all ML models
            try:
                self.ml_engine.train_demand_forecasting_models()
                self.ml_engine.train_pricing_models()
                self.ml_engine.train_route_recommendation_models()
                self.ml_engine.train_weather_impact_models()

                # Save models
                self.ml_engine.save_models()

                # Generate ML predictions for Power BI
                predictions_df = self.ml_engine.generate_batch_predictions()

                # Create ML-enhanced insights
                ml_insights = self._create_ml_enhanced_insights()

                print("✅ ML models trained and predictions generated")

            except Exception as e:
                print(f"⚠️ ML training error: {e}")
                ml_insights = {}
        else:
            ml_insights = {}
            predictions_df = None

        # 3. Generate comprehensive reports
        print("\n📋 Phase 3: Comprehensive Reporting")

        # Combined insights
        bi_insights = self.bi_analyzer.generate_power_bi_insights()
        combined_insights = {**bi_insights, **ml_insights}

        # Save all results
        self._save_comprehensive_results(combined_insights, predictions_df)

        # Create executive summary
        self._create_executive_summary(combined_insights)

        return combined_insights

    def _create_ml_enhanced_insights(self):
        """Create insights enhanced with ML predictions"""
        if not self.ml_engine:
            return {}

        # Get model performance metrics
        model_summary = self.ml_engine.get_model_summary()

        # Test predictions for key scenarios
        test_predictions = {}

        try:
            # Peak hour rain scenario
            peak_rain_demand = self.ml_engine.predict_demand(
                hour=18, day_of_week=1, month=6,
                is_weekend=False, weather_condition='Rain'
            )

            # Clear weather pricing
            clear_pricing = self.ml_engine.predict_pricing(
                hour=18, day_of_week=1, distance_km=5.0,
                weather_condition='Clear', vehicle_type='auto'
            )

            # Rain weather pricing
            rain_pricing = self.ml_engine.predict_pricing(
                hour=18, day_of_week=1, distance_km=5.0,
                weather_condition='Rain', vehicle_type='auto'
            )

            test_predictions = {
                'peak_rain_demand': peak_rain_demand,
                'clear_pricing': clear_pricing,
                'rain_pricing': rain_pricing,
                'predicted_rain_premium': (rain_pricing['Ensemble'] / clear_pricing['Ensemble'] - 1) * 100
            }

        except Exception as e:
            print(f"⚠️ Prediction test error: {e}")

        return {
            'ml_model_performance': model_summary,
            'ml_predictions': test_predictions,
            'ml_features_count': {
                'demand_features': len(self.ml_engine.demand_features),
                'pricing_features': len(self.ml_engine.pricing_features),
                'route_features': len(self.ml_engine.route_features),
                'weather_features': len(self.ml_engine.weather_impact_features)
            }
        }

    def _save_comprehensive_results(self, insights, predictions_df):
        """Save all results including ML predictions"""
        import json
        from pathlib import Path

        # Create output directories
        output_dir = "analysis/comprehensive_analysis/"
        power_bi_dir = "power_bi/data_model/"

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(power_bi_dir).mkdir(parents=True, exist_ok=True)

        # Save comprehensive insights
        with open(f"{output_dir}comprehensive_insights.json", 'w') as f:
            json.dump(insights, f, indent=2, default=str)

        # Save ML predictions for Power BI if available
        if predictions_df is not None:
            predictions_df.to_excel(f"{power_bi_dir}ml_predictions.xlsx", index=False)
            print(f"✅ ML predictions saved to {power_bi_dir}ml_predictions.xlsx")

        # Save geospatial features if available
        if self.geo_engine:
            try:
                self.geo_engine.export_location_data_for_powerbi(self.ml_engine.df)
                print("✅ Geospatial features exported for Power BI")
            except Exception:
                pass

        # Save weather data if available
        if self.weather_api:
            try:
                self.weather_api.save_forecast_data()
                print("✅ Weather forecast data saved for Power BI")
            except Exception:
                pass

        print(f"✅ Comprehensive results saved to {output_dir}")

    def _create_executive_summary(self, insights):
        """Create executive summary with ML enhancements"""
        print("\n" + "=" * 80)
        print("🎯 COMPREHENSIVE EXECUTIVE SUMMARY - RAPIDO TRANSPORTATION")
        print("=" * 80)

        # BI Metrics
        if 'key_metrics' in insights:
            print("\n📊 BUSINESS INTELLIGENCE METRICS:")
            for metric, value in insights['key_metrics'].items():
                print(f"  • {metric.replace('_', ' ').title()}: {value}")

        # ML Enhancements
        if 'ml_model_performance' in insights:
            print("\n🤖 MACHINE LEARNING ENHANCEMENTS:")
            ml_perf = insights['ml_model_performance']
            print(f"  • Models Trained: {ml_perf.get('models_trained', 0)}")
            print(f"  • Model Types: {', '.join(ml_perf.get('model_types', []))}")

            if 'best_models' in ml_perf:
                print("  • Best Performing Models:")
                for model_type, performance in ml_perf['best_models'].items():
                    if 'R2' in performance:
                        print(f"    - {model_type}: {performance['model']} (R² = {performance['R2']:.3f})")
                    elif 'Accuracy' in performance:
                        print(f"    - {model_type}: {performance['model']} (Accuracy = {performance['Accuracy']:.3f})")

        # Predictions
        if 'ml_predictions' in insights and insights['ml_predictions']:
            print("\n🔮 PREDICTIVE INSIGHTS:")
            pred = insights['ml_predictions']
            if 'predicted_rain_premium' in pred:
                print(f"  • Predicted Rain Premium: {pred['predicted_rain_premium']:.1f}%")

        # Time Patterns
        if 'time_patterns' in insights:
            print("\n⏰ TIME PATTERN INSIGHTS:")
            for pattern, value in insights['time_patterns'].items():
                print(f"  • {pattern.replace('_', ' ').title()}: {value}")

        # Weather Insights
        if 'weather_insights' in insights:
            print("\n🌦️ WEATHER IMPACT INSIGHTS:")
            for weather, value in insights['weather_insights'].items():
                print(f"  • {weather.replace('_', ' ').title()}: {value}")

        print("\n" + "=" * 80)
        print("✅ COMPREHENSIVE ANALYSIS COMPLETE")
        print("📁 Results available in: analysis/comprehensive_analysis/")
        print("📊 Power BI files ready in: power_bi/data_model/")
        print("=" * 80)


def main_ml_integration():
    """Main function for ML-integrated analysis"""
    print("🚀 Starting Comprehensive ML + BI Analysis for Rapido Transportation")
    print("=" * 80)

    # Initialize comprehensive analyzer
    analyzer = RapidoMLBIIntegration("Dataset.csv")

    # Run complete analysis
    results = analyzer.run_complete_analysis()

    print("\n🎉 ML-Enhanced Analysis Complete! Check the output directories for results.")
    return analyzer, results


if __name__ == "__main__":
    # Run traditional BI analysis first
    analyzer, summary, power_bi_insights = main()

    # Then run ML-enhanced analysis if requested
    try:
        ml_analyzer, ml_results = main_ml_integration()
        print("\n✅ Both BI and ML analyses completed successfully!")
    except Exception as e:
        print(f"\n⚠️ ML analysis failed (BI analysis successful): {e}")
        print("📊 Traditional BI results are available in analysis/business_insights/")

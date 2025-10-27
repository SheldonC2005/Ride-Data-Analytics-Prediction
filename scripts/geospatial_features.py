"""
Geospatial Features Enhancement for Rapido Transportation ML
Author: BI Project Team
Date: October 14, 2025

This module adds geospatial intelligence to enhance ML predictions
using existing from/to location data without external mapping APIs.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from math import radians, cos, sin, asin, sqrt, atan2
import re
from collections import defaultdict

class GeospatialFeatureEngine:
    """
    Geospatial feature engineering using existing location data
    """
    
    def __init__(self):
        """Initialize geospatial engine"""
        self.location_coordinates = {}
        self.location_clusters = {}
        self.route_patterns = {}
        self.vit_coordinates = (12.9685, 79.1550)  # VIT University approximate coordinates
        
        print("📍 Geospatial Feature Engine initialized")
    
    def process_dataset(self, df):
        """
        Process dataset to extract and enhance geospatial features
        
        Args:
            df: DataFrame with ride data
            
        Returns:
            Enhanced DataFrame with geospatial features
        """
        print("🗺️ Processing geospatial features...")
        
        # Extract location patterns
        self._extract_location_patterns(df)
        
        # Estimate coordinates for locations
        self._estimate_location_coordinates(df)
        
        # Add distance and direction features
        df = self._add_distance_features(df)
        
        # Add location clustering features
        df = self._add_clustering_features(df)
        
        # Add route complexity features
        df = self._add_route_complexity_features(df)
        
        # Add accessibility features
        df = self._add_accessibility_features(df)
        
        print("✅ Geospatial features processed")
        return df
    
    def _extract_location_patterns(self, df):
        """Extract location patterns and frequencies"""
        # Get all unique locations
        from_locations = df['from_location'].value_counts()
        to_locations = df['to_location'].value_counts()
        
        # Combine and rank locations
        all_locations = defaultdict(int)
        
        for loc, count in from_locations.items():
            all_locations[loc] += count
        
        for loc, count in to_locations.items():
            all_locations[loc] += count
        
        # Store location rankings
        self.location_rankings = dict(all_locations)
        
        # Identify major hubs (top 20% of locations by frequency)
        total_locations = len(all_locations)
        hub_threshold = int(total_locations * 0.2)
        
        sorted_locations = sorted(all_locations.items(), key=lambda x: x[1], reverse=True)
        self.major_hubs = set([loc for loc, _ in sorted_locations[:hub_threshold]])
        
        print(f"Identified {len(self.major_hubs)} major transportation hubs")
    
    def _estimate_location_coordinates(self, df):
        """Estimate coordinates for locations based on patterns and known landmarks"""
        # Known landmark coordinates in Vellore
        known_coordinates = {
            'VIT University': (12.9685, 79.1550),
            'VIT': (12.9685, 79.1550),
            'Vellore Railway Station': (12.9249, 79.1378),
            'Railway Station': (12.9249, 79.1378),
            'CMC Hospital': (12.9259, 79.1353),
            'Green Circle': (12.9200, 79.1400),  # Estimated
            'Katpadi': (12.9698, 79.2031),
            'Bagayam': (12.9150, 79.1320),  # Estimated
            'Arcot Road': (12.9100, 79.1500),  # Estimated
            'Vellore Fort': (12.9206, 79.1348),
        }
        
        # Add known coordinates
        self.location_coordinates.update(known_coordinates)
        
        # Estimate coordinates for unknown locations
        unknown_locations = set()
        
        for _, row in df.iterrows():
            from_loc = row['from_location']
            to_loc = row['to_location']
            
            if from_loc not in self.location_coordinates:
                unknown_locations.add(from_loc)
            if to_loc not in self.location_coordinates:
                unknown_locations.add(to_loc)
        
        # Estimate coordinates for unknown locations
        self._estimate_unknown_coordinates(unknown_locations, df)
        
        print(f"Coordinates available for {len(self.location_coordinates)} locations")
    
    def _estimate_unknown_coordinates(self, unknown_locations, df):
        """Estimate coordinates for unknown locations"""
        # Vellore city center as reference
        vellore_center = (12.9165, 79.1325)
        
        for location in unknown_locations:
            # Check if location contains keywords for estimation
            location_lower = location.lower()
            
            if 'vit' in location_lower:
                # Near VIT University
                lat_offset = np.random.uniform(-0.005, 0.005)
                lon_offset = np.random.uniform(-0.005, 0.005)
                estimated_coord = (
                    self.vit_coordinates[0] + lat_offset,
                    self.vit_coordinates[1] + lon_offset
                )
            elif 'railway' in location_lower or 'station' in location_lower:
                # Near railway station
                lat_offset = np.random.uniform(-0.003, 0.003)
                lon_offset = np.random.uniform(-0.003, 0.003)
                estimated_coord = (
                    12.9249 + lat_offset,
                    79.1378 + lon_offset
                )
            elif 'hospital' in location_lower or 'cmc' in location_lower:
                # Near CMC Hospital
                lat_offset = np.random.uniform(-0.003, 0.003)
                lon_offset = np.random.uniform(-0.003, 0.003)
                estimated_coord = (
                    12.9259 + lat_offset,
                    79.1353 + lon_offset
                )
            else:
                # Random location within Vellore bounds
                # Vellore is roughly within these bounds
                lat = np.random.uniform(12.90, 12.98)
                lon = np.random.uniform(79.10, 79.20)
                estimated_coord = (lat, lon)
            
            self.location_coordinates[location] = estimated_coord
    
    def _calculate_haversine_distance(self, coord1, coord2):
        """Calculate haversine distance between two coordinates"""
        lat1, lon1 = radians(coord1[0]), radians(coord1[1])
        lat2, lon2 = radians(coord2[0]), radians(coord2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Earth radius in kilometers
        r = 6371
        return c * r
    
    def _calculate_bearing(self, coord1, coord2):
        """Calculate bearing (direction) between two coordinates"""
        lat1, lon1 = radians(coord1[0]), radians(coord1[1])
        lat2, lon2 = radians(coord2[0]), radians(coord2[1])
        
        dlon = lon2 - lon1
        
        y = sin(dlon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        
        bearing = atan2(y, x)
        bearing = (bearing * 180 / np.pi + 360) % 360
        
        return bearing
    
    def _add_distance_features(self, df):
        """Add distance and direction features"""
        print("📏 Adding distance and direction features...")
        
        # Calculate estimated distances
        estimated_distances = []
        bearings = []
        
        for _, row in df.iterrows():
            from_coord = self.location_coordinates.get(row['from_location'])
            to_coord = self.location_coordinates.get(row['to_location'])
            
            if from_coord and to_coord:
                # Calculate estimated distance
                est_distance = self._calculate_haversine_distance(from_coord, to_coord)
                bearing = self._calculate_bearing(from_coord, to_coord)
            else:
                # Use existing distance as fallback
                est_distance = row.get('distance_km', 5.0)
                bearing = np.random.uniform(0, 360)  # Random bearing
            
            estimated_distances.append(est_distance)
            bearings.append(bearing)
        
        df['estimated_distance_km'] = estimated_distances
        df['route_bearing'] = bearings
        
        # Distance categories
        df['distance_category_detailed'] = pd.cut(
            df['estimated_distance_km'],
            bins=[0, 1, 3, 5, 8, 12, float('inf')],
            labels=['hyperlocal', 'local', 'short', 'medium', 'long', 'intercity']
        )
        
        # Direction categories (8-point compass)
        df['direction_category'] = pd.cut(
            df['route_bearing'],
            bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
            labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        )
        
        # Distance accuracy (how close estimated vs actual)
        if 'distance_km' in df.columns:
            df['distance_accuracy'] = 1 - abs(df['estimated_distance_km'] - df['distance_km']) / (df['distance_km'] + 1e-6)
            df['distance_accuracy'] = df['distance_accuracy'].clip(0, 1)
        
        return df
    
    def _add_clustering_features(self, df):
        """Add location clustering features"""
        print("🎯 Adding clustering features...")
        
        # Hub classification
        df['from_is_major_hub'] = df['from_location'].isin(self.major_hubs).astype(int)
        df['to_is_major_hub'] = df['to_location'].isin(self.major_hubs).astype(int)
        df['involves_major_hub'] = ((df['from_is_major_hub'] == 1) | (df['to_is_major_hub'] == 1)).astype(int)
        
        # VIT involvement (detailed)
        df['from_is_vit'] = df['from_location'].str.contains('VIT', case=False, na=False).astype(int)
        df['to_is_vit'] = df['to_location'].str.contains('VIT', case=False, na=False).astype(int)
        df['vit_to_vit'] = ((df['from_is_vit'] == 1) & (df['to_is_vit'] == 1)).astype(int)
        df['vit_outbound'] = ((df['from_is_vit'] == 1) & (df['to_is_vit'] == 0)).astype(int)
        df['vit_inbound'] = ((df['from_is_vit'] == 0) & (df['to_is_vit'] == 1)).astype(int)
        
        # Location popularity scores
        df['from_location_popularity'] = df['from_location'].map(self.location_rankings).fillna(0)
        df['to_location_popularity'] = df['to_location'].map(self.location_rankings).fillna(0)
        df['route_popularity_score'] = (df['from_location_popularity'] + df['to_location_popularity']) / 2
        
        # Location type classification
        df['from_location_type'] = df['from_location'].apply(self._classify_location_type)
        df['to_location_type'] = df['to_location'].apply(self._classify_location_type)
        
        return df
    
    def _classify_location_type(self, location):
        """Classify location type based on name patterns"""
        location_lower = location.lower()
        
        if 'vit' in location_lower or 'university' in location_lower:
            return 'educational'
        elif 'hospital' in location_lower or 'cmc' in location_lower:
            return 'medical'
        elif 'railway' in location_lower or 'station' in location_lower:
            return 'transport'
        elif 'circle' in location_lower or 'junction' in location_lower:
            return 'junction'
        elif 'mall' in location_lower or 'market' in location_lower:
            return 'commercial'
        elif 'fort' in location_lower or 'temple' in location_lower:
            return 'heritage'
        else:
            return 'residential'
    
    def _add_route_complexity_features(self, df):
        """Add route complexity and accessibility features"""
        print("🛣️ Adding route complexity features...")
        
        # Route frequency and patterns
        route_counts = df.groupby('route').size()
        reverse_routes = {}
        
        for route in route_counts.index:
            from_loc, to_loc = route.split(' → ')
            reverse_route = f"{to_loc} → {from_loc}"
            reverse_routes[route] = reverse_route
        
        df['route_frequency'] = df['route'].map(route_counts)
        df['reverse_route'] = df['route'].map(reverse_routes)
        df['reverse_route_exists'] = df['reverse_route'].isin(route_counts.index).astype(int)
        
        # Route complexity based on multiple factors
        complexity_scores = []
        
        for _, row in df.iterrows():
            score = 0
            
            # Distance complexity
            if row['estimated_distance_km'] > 10:
                score += 3
            elif row['estimated_distance_km'] > 5:
                score += 2
            else:
                score += 1
            
            # Hub involvement complexity
            if row['involves_major_hub'] == 1:
                score += 2
            
            # VIT involvement (high traffic)
            if row['from_is_vit'] == 1 or row['to_is_vit'] == 1:
                score += 2
            
            # Location type complexity
            if row['from_location_type'] == 'transport' or row['to_location_type'] == 'transport':
                score += 1  # Transport hubs can be complex
            
            complexity_scores.append(min(score, 10))  # Cap at 10
        
        df['route_complexity_score'] = complexity_scores
        
        # Network centrality (simplified)
        df['from_centrality'] = df['from_location_popularity'] / df['from_location_popularity'].max()
        df['to_centrality'] = df['to_location_popularity'] / df['to_location_popularity'].max()
        df['route_centrality'] = (df['from_centrality'] + df['to_centrality']) / 2
        
        return df
    
    def _add_accessibility_features(self, df):
        """Add accessibility and service coverage features"""
        print("♿ Adding accessibility features...")
        
        # Distance to VIT (important hub)
        vit_distances_from = []
        vit_distances_to = []
        
        for _, row in df.iterrows():
            from_coord = self.location_coordinates.get(row['from_location'])
            to_coord = self.location_coordinates.get(row['to_location'])
            
            if from_coord:
                vit_dist_from = self._calculate_haversine_distance(from_coord, self.vit_coordinates)
            else:
                vit_dist_from = 5.0  # Default
            
            if to_coord:
                vit_dist_to = self._calculate_haversine_distance(to_coord, self.vit_coordinates)
            else:
                vit_dist_to = 5.0  # Default
            
            vit_distances_from.append(vit_dist_from)
            vit_distances_to.append(vit_dist_to)
        
        df['distance_from_vit'] = vit_distances_from
        df['distance_to_vit'] = vit_distances_to
        df['min_distance_to_vit'] = df[['distance_from_vit', 'distance_to_vit']].min(axis=1)
        
        # Service area classification
        df['service_area'] = pd.cut(
            df['min_distance_to_vit'],
            bins=[0, 2, 5, 10, float('inf')],
            labels=['core', 'extended', 'suburban', 'outer']
        )
        
        # Route efficiency score (inverse of detour factor)
        if 'distance_km' in df.columns:
            df['route_efficiency'] = df['distance_km'] / (df['estimated_distance_km'] + 1e-6)
            df['route_efficiency'] = df['route_efficiency'].clip(0.5, 2.0)  # Reasonable bounds
        else:
            df['route_efficiency'] = 1.0  # Default
        
        # Cross-town vs local classification
        df['is_cross_town'] = (df['estimated_distance_km'] > 8).astype(int)
        df['is_local'] = (df['estimated_distance_km'] <= 3).astype(int)
        
        return df
    
    def get_location_insights(self, df):
        """Generate location-based business insights"""
        insights = {
            'top_origins': df['from_location'].value_counts().head(10).to_dict(),
            'top_destinations': df['to_location'].value_counts().head(10).to_dict(),
            'busiest_routes': df['route'].value_counts().head(10).to_dict(),
            'location_types': {
                'educational_trips': len(df[(df['from_location_type'] == 'educational') | 
                                          (df['to_location_type'] == 'educational')]),
                'medical_trips': len(df[(df['from_location_type'] == 'medical') | 
                                      (df['to_location_type'] == 'medical')]),
                'transport_hub_trips': len(df[(df['from_location_type'] == 'transport') | 
                                            (df['to_location_type'] == 'transport')])
            },
            'service_coverage': {
                'core_area_trips': len(df[df['service_area'] == 'core']),
                'extended_area_trips': len(df[df['service_area'] == 'extended']),
                'suburban_trips': len(df[df['service_area'] == 'suburban']),
                'outer_area_trips': len(df[df['service_area'] == 'outer'])
            },
            'vit_dependency': {
                'total_vit_trips': len(df[(df['from_is_vit'] == 1) | (df['to_is_vit'] == 1)]),
                'vit_outbound': len(df[df['vit_outbound'] == 1]),
                'vit_inbound': len(df[df['vit_inbound'] == 1]),
                'vit_percentage': (len(df[(df['from_is_vit'] == 1) | (df['to_is_vit'] == 1)]) / len(df)) * 100
            }
        }
        
        return insights
    
    def generate_geospatial_report(self, df, output_path="analysis/geospatial_analysis.json"):
        """Generate comprehensive geospatial analysis report"""
        insights = self.get_location_insights(df)
        
        # Add statistical summaries
        geospatial_stats = {
            'distance_statistics': {
                'avg_estimated_distance': df['estimated_distance_km'].mean(),
                'max_distance': df['estimated_distance_km'].max(),
                'min_distance': df['estimated_distance_km'].min(),
                'distance_std': df['estimated_distance_km'].std()
            },
            'complexity_analysis': {
                'avg_route_complexity': df['route_complexity_score'].mean(),
                'high_complexity_routes': len(df[df['route_complexity_score'] >= 7]),
                'simple_routes': len(df[df['route_complexity_score'] <= 3])
            },
            'centrality_metrics': {
                'avg_route_centrality': df['route_centrality'].mean(),
                'high_centrality_trips': len(df[df['route_centrality'] >= 0.8])
            }
        }
        
        # Combine all insights
        report = {
            'location_insights': insights,
            'geospatial_statistics': geospatial_stats,
            'major_hubs': list(self.major_hubs),
            'total_locations': len(self.location_coordinates),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save report
        import json
        from pathlib import Path
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Geospatial analysis report saved to {output_path}")
        return report
    
    def export_location_data_for_powerbi(self, df, output_path="power_bi/data_model/location_features.xlsx"):
        """Export location features for Power BI integration"""
        # Create location dimension table
        locations = set(df['from_location'].tolist() + df['to_location'].tolist())
        
        location_dim = []
        for location in locations:
            coord = self.location_coordinates.get(location, (0, 0))
            popularity = self.location_rankings.get(location, 0)
            location_type = self._classify_location_type(location)
            
            is_hub = location in self.major_hubs
            is_vit = 'vit' in location.lower()
            
            distance_to_vit = self._calculate_haversine_distance(coord, self.vit_coordinates) if coord != (0, 0) else 0
            
            location_dim.append({
                'location_name': location,
                'latitude': coord[0],
                'longitude': coord[1],
                'popularity_score': popularity,
                'location_type': location_type,
                'is_major_hub': int(is_hub),
                'is_vit_location': int(is_vit),
                'distance_to_vit_km': distance_to_vit,
                'service_area': 'core' if distance_to_vit <= 2 else 'extended' if distance_to_vit <= 5 else 'suburban' if distance_to_vit <= 10 else 'outer'
            })
        
        location_df = pd.DataFrame(location_dim)
        
        # Create route features summary
        route_features = df.groupby('route').agg({
            'route_frequency': 'first',
            'route_complexity_score': 'first',
            'route_centrality': 'first',
            'estimated_distance_km': 'first',
            'route_efficiency': 'first',
            'final_price': 'mean'
        }).reset_index()
        
        route_features.rename(columns={'final_price': 'avg_route_price'}, inplace=True)
        
        # Save to Excel
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            location_df.to_excel(writer, sheet_name='LocationDimension', index=False)
            route_features.to_excel(writer, sheet_name='RouteFeatures', index=False)
        
        print(f"✅ Location features exported to {output_path}")
        return location_df, route_features

# Example usage
def main():
    """Test geospatial feature engine"""
    print("📍 Testing Geospatial Feature Engine...")
    
    # Load sample data (replace with actual data path)
    try:
        df = pd.read_csv("Dataset.csv")
        print(f"Loaded dataset with {len(df)} records")
    except:
        print("Dataset not found - creating sample data")
        # Create sample data for testing
        df = pd.DataFrame({
            'from_location': ['VIT University', 'Katpadi', 'Green Circle'] * 100,
            'to_location': ['Railway Station', 'CMC Hospital', 'VIT University'] * 100,
            'distance_km': np.random.uniform(2, 15, 300),
            'final_price': np.random.uniform(50, 300, 300)
        })
        df['route'] = df['from_location'] + ' → ' + df['to_location']
    
    # Initialize geospatial engine
    geo_engine = GeospatialFeatureEngine()
    
    # Process geospatial features
    enhanced_df = geo_engine.process_dataset(df)
    
    # Generate insights
    insights = geo_engine.get_location_insights(enhanced_df)
    print(f"Generated insights for {len(insights)} categories")
    
    # Generate report
    report = geo_engine.generate_geospatial_report(enhanced_df)
    
    # Export for Power BI
    geo_engine.export_location_data_for_powerbi(enhanced_df)
    
    print("✅ Geospatial feature engine test complete!")
    print(f"Added {len([col for col in enhanced_df.columns if 'location' in col or 'distance' in col or 'route' in col])} geospatial features")

if __name__ == "__main__":
    main()
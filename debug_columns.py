import sys
sys.path.insert(0, 'scripts')
from ml_prediction_engine import RapidoMLEngine  # type: ignore

# Initialize and load data
engine = RapidoMLEngine('Dataset.csv')
engine.load_and_prepare_data()

# Check for duplicate columns
cols = engine.df.columns.tolist()
dup_cols = engine.df.columns[engine.df.columns.duplicated()].tolist()

print("\n=== COLUMNS AFTER FEATURE ENGINEERING ===")
print(f"Total columns: {len(cols)}")
print(f"Unique columns: {len(set(cols))}")
print(f"\nDuplicate columns: {dup_cols}")
print(f"Hour count: {cols.count('hour')}")

# Check prepare_ml_features
engine.prepare_ml_features()
cols2 = engine.df.columns.tolist()
dup_cols2 = engine.df.columns[engine.df.columns.duplicated()].tolist()

print("\n=== COLUMNS AFTER prepare_ml_features ===")
print(f"Total columns: {len(cols2)}")
print(f"Unique columns: {len(set(cols2))}")
print(f"\nDuplicate columns: {dup_cols2}")
print(f"Hour count: {cols2.count('hour')}")

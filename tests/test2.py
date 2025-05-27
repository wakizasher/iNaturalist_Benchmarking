import pandas as pd

# Load your CSV and check the species names
df = pd.read_csv(r"/data/ground_truth_filtered.csv")
print("Unique species in your CSV:")
print(df['flower_name'].unique())

print("\nFirst 5 rows:")
print(df[['file_path', 'flower_name']].head())
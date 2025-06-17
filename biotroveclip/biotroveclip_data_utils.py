"""
📁 Data Utilities Module
Functions for loading and validating ground truth data

Author: Nikita Gavrilov
"""

import pandas as pd
import os


def load_and_validate_ground_truth(csv_path, dataset_name):
    """
    📁 Load and validate ground truth CSV file

    Args:
        csv_path (str): Path to CSV file
        dataset_name (str): Name of dataset for logging

    Returns:
        pd.DataFrame: Validated DataFrame

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    print(f"📊 Loading {dataset_name} ground truth data...")

    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ CSV file not found: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Validate columns
    required_columns = ['file_path', 'species']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"❌ Missing columns in {dataset_name}: {missing_columns}")

    # Check for missing values
    if df[required_columns].isnull().any().any():
        print(f"⚠️ Warning: Found missing values in {dataset_name}")
        df = df.dropna(subset=required_columns)
        print(f"📊 After removing missing values: {len(df)} images")

    # Validate image files exist with simple progress
    print(f"🔍 Validating image files exist...")
    missing_files = []

    for idx, row in df.iterrows():
        if not os.path.exists(row['file_path']):
            missing_files.append(row['file_path'])

        # Show progress every 500 files
        if (idx + 1) % 500 == 0 or (idx + 1) == len(df):
            print(f"   ✅ Checked {idx + 1}/{len(df)} file paths")

    if missing_files:
        print(f"⚠️ Warning: {len(missing_files)} image files not found in {dataset_name}")
        print("📋 First few missing files:")
        for file in missing_files[:5]:
            print(f"   - {file}")

        # Remove entries with missing files
        df = df[df['file_path'].apply(os.path.exists)]
        print(f"📊 After removing missing files: {len(df)} images")

    # Show dataset statistics
    print(f"📈 {dataset_name}: {len(df)} valid images")
    print(f"🌸 Species distribution:")
    species_counts = df['species'].value_counts()
    for species, count in species_counts.items():
        print(f"   - {species}: {count} images")

    return df


def validate_file_paths(file_paths):
    """
    🔍 Validate that all file paths exist

    Args:
        file_paths (list): List of file paths to check

    Returns:
        tuple: (valid_paths, missing_paths)
    """
    valid_paths = []
    missing_paths = []

    for path in file_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            missing_paths.append(path)

    return valid_paths, missing_paths


def get_species_distribution(df, column_name='species'):
    """
    📊 Get species distribution from DataFrame

    Args:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Column containing species names

    Returns:
        pd.Series: Species counts
    """
    return df[column_name].value_counts()


def sample_balanced_data(df, n_samples_per_species, species_column='species'):
    """
    🎲 Sample balanced data from DataFrame

    Args:
        df (pd.DataFrame): Input DataFrame
        n_samples_per_species (int): Number of samples per species
        species_column (str): Column containing species names

    Returns:
        pd.DataFrame: Balanced sample
    """
    sampled_dfs = []

    for species in df[species_column].unique():
        species_df = df[df[species_column] == species]
        if len(species_df) >= n_samples_per_species:
            sampled = species_df.sample(n=n_samples_per_species, random_state=42)
        else:
            sampled = species_df  # Use all available if less than requested
        sampled_dfs.append(sampled)

    return pd.concat(sampled_dfs, ignore_index=True)
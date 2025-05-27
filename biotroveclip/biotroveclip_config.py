"""
⚙️ BioTroveCLIP Configuration Module
Central configuration for BioTroveCLIP flower identification experiment

Author: Nikita Gavrilov
"""

import os
from datetime import datetime

# 📁 File Paths - UPDATE THESE TO YOUR ACTUAL PATHS
FILTERED_CSV = "C:\\Users\\Nikita\\Projects\\ZeroShot\\data\\ground_truth_filtered.csv"
UNFILTERED_CSV = "C:\\Users\\Nikita\\Projects\\ZeroShot\\data\\ground_truth_original.csv"

# 🌸 Flower Species Configuration
FLOWER_SPECIES = [
    "a photo of Bellis perennis",
    "a photo of Matricaria chamomilla",
    "a photo of Leucanthemum vulgare"
]

SPECIES_NAMES = [
    "Bellis perennis",
    "Matricaria chamomilla",
    "Leucanthemum vulgare"
]

# 🔧 Model Configuration
MODEL_NAME = 'hf-hub:BGLab/BioTrove-CLIP'
DEVICE = "cpu"
NUM_THREADS = 12  # For Ryzen 5 7600X (6 cores × 2 threads)

# 📊 Processing Configuration
PROGRESS_UPDATE_INTERVAL = 100  # Show progress every N images
DEBUG_SAMPLE_SIZE = 3  # Number of images to show in debug mode
FILE_CHECK_INTERVAL = 500  # Show file validation progress every N files

# 📁 Output Configuration
OUTPUT_DIR = "biotroveclip_results"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

def get_output_filename(base_name, extension="json"):
    """Generate timestamped output filename"""
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    return f"{base_name}_{timestamp}.{extension}"

def create_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Created output directory: {OUTPUT_DIR}")

# 📈 Analysis Configuration
CONFIDENCE_THRESHOLDS = {
    'high': 0.9,
    'medium': 0.7,
    'low': 0.5
}

STATISTICAL_SIGNIFICANCE_ALPHA = 0.05

# 🎨 Visualization Configuration
FIGURE_SIZE = (12, 8)
DPI = 300
COLOR_PALETTE = ['#2E8B57', '#FF6B6B', '#4ECDC4']  # Colors for the 3 species

# 🔍 Validation Configuration
BOOTSTRAP_SAMPLES = 1000
CROSS_VALIDATION_FOLDS = 5

# 📊 Metrics Configuration
METRICS_TO_CALCULATE = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'confidence_stats',
    'confusion_matrix'
]

# Print configuration summary
def print_config_summary():
    """Print configuration summary for verification"""
    print("⚙️ BIOTROVECLIP EXPERIMENT CONFIGURATION")
    print("=" * 50)
    print(f"📁 Filtered CSV: {FILTERED_CSV}")
    print(f"📁 Unfiltered CSV: {UNFILTERED_CSV}")
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}")
    print(f"🔧 Threads: {NUM_THREADS}")
    print(f"🌸 Species: {len(SPECIES_NAMES)} species")
    print(f"📊 Output directory: {OUTPUT_DIR}")
    print("=" * 50)

# Validation functions
def validate_config():
    """Validate configuration settings"""
    errors = []

    # Check if CSV files exist
    if not os.path.exists(FILTERED_CSV):
        errors.append(f"Filtered CSV not found: {FILTERED_CSV}")

    if not os.path.exists(UNFILTERED_CSV):
        errors.append(f"Unfiltered CSV not found: {UNFILTERED_CSV}")

    # Check species consistency
    if len(FLOWER_SPECIES) != len(SPECIES_NAMES):
        errors.append("FLOWER_SPECIES and SPECIES_NAMES must have same length")

    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"   - {error}")
        return False

    return True
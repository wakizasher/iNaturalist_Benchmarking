"""
⚙️ Configuration Module - Optimized for RTX 3060 Laptop
Central configuration for BioCLIP flower identification experiment
AMD Ryzen 5 5600H + RTX 3060 Laptop + 16GB RAM

Author: Nikita Gavrilov
"""

import os
import torch
import psutil
from datetime import datetime

# 📁 File Paths - YOUR ACTUAL PATHS
FILTERED_CSV = "C:\\Users\\nikit\\Pycharm\\iNaturalist_Benchmarking\\extended_data\\vlm_q3_bad_images456.csv"
UNFILTERED_CSV = "C:\\Users\\nikit\\Pycharm\\iNaturalist_Benchmarking\\extended_data\seed456_base.csv"
  
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

# 🔧 Hardware-Optimized Model Configuration
def get_optimal_device_config():
    """
    🎯 Automatically detect and configure optimal device settings for your laptop
    """
    # Check CUDA availability for RTX 3060
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        device = "cuda"
        # RTX 3060 Laptop has 6GB VRAM - perfect for BioCLIP
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        num_threads = 6  # Reduce CPU threads when using GPU (Ryzen 5 5600H optimization)
        print(f"🎮 GPU detected: {gpu_name}")
        print(f"💾 GPU memory: {gpu_memory:.1f} GB")
        print(f"⚡ GPU acceleration enabled!")
    else:
        device = "cpu"
        num_threads = 12  # Full CPU power: Ryzen 5 5600H (6 cores × 2 threads)
        print(f"💻 CUDA not available, using optimized CPU mode")

    return device, num_threads

# Get optimal configuration for your hardware
DEVICE, NUM_THREADS = get_optimal_device_config()

# Model Configuration
MODEL_NAME = 'hf-hub:BGLab/BioTrove-CLIP'

# 📊 Processing Configuration - Laptop Optimized
PROGRESS_UPDATE_INTERVAL = 50   # More frequent updates (every 50 images) for laptop
DEBUG_SAMPLE_SIZE = 3
FILE_CHECK_INTERVAL = 200       # More frequent file validation updates

# 🧠 Memory Management for RTX 3060 Laptop
BATCH_SIZE = 1                  # Process one image at a time (safest for laptop)
ENABLE_MIXED_PRECISION = True   # Use half precision to save VRAM on RTX 3060
CLEAR_CACHE_FREQUENCY = 100     # Clear GPU cache every 100 images to prevent memory issues

# 📁 Output Configuration
OUTPUT_DIR = "../results"
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

# 🎨 Visualization Configuration - Laptop Screen Optimized
FIGURE_SIZE = (10, 6)  # Slightly smaller for laptop screen
DPI = 150              # Lower DPI for faster rendering on laptop
COLOR_PALETTE = ['#2E8B57', '#FF6B6B', '#4ECDC4']  # Colors for the 3 species

# 🔍 Validation Configuration - Laptop Performance Optimized
BOOTSTRAP_SAMPLES = 500    # Reduced for faster computation on laptop
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

# 🖥️ Laptop-Specific Performance Settings
def setup_laptop_optimizations():
    """
    ⚡ Configure laptop-specific optimizations for RTX 3060 + Ryzen 5 5600H
    """
    print("🖥️ LAPTOP OPTIMIZATION SETUP")
    print("=" * 40)

    # CPU optimizations for Ryzen 5 5600H
    torch.set_num_threads(NUM_THREADS)
    torch.backends.mkldnn.enabled = True

    # GPU optimizations for RTX 3060 (if available)
    if DEVICE == "cuda":
        # Enable optimizations for RTX 3060
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed

        # Memory management for 6GB VRAM
        if ENABLE_MIXED_PRECISION:
            print("🔄 Mixed precision enabled - saves ~50% GPU memory!")

        print(f"🎮 GPU optimizations enabled for {torch.cuda.get_device_name(0)}")
        print(f"💾 Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    print(f"🔧 CPU threads: {NUM_THREADS} (optimized for Ryzen 5 5600H)")
    print(f"💻 Device: {DEVICE}")
    print(f"⚡ Laptop optimizations applied!")

# Print configuration summary
def print_config_summary():
    """Print configuration summary for verification"""
    print("⚙️ LAPTOP-OPTIMIZED EXPERIMENT CONFIGURATION")
    print("=" * 55)
    print(f"💻 Hardware: AMD Ryzen 5 5600H + RTX 3060 Laptop (16GB RAM)")
    print(f"📁 Filtered CSV: {FILTERED_CSV}")
    print(f"📁 Unfiltered CSV: {UNFILTERED_CSV}")
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}")
    print(f"🔧 CPU Threads: {NUM_THREADS}")

    if DEVICE == "cuda":
        print(f"🧠 Mixed Precision: {ENABLE_MIXED_PRECISION}")
        print(f"🎮 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    print(f"🌸 Species: {len(SPECIES_NAMES)} species")
    print(f"📊 Output directory: {OUTPUT_DIR}")
    print(f"📈 Progress updates every {PROGRESS_UPDATE_INTERVAL} images")
    print("=" * 55)

# Enhanced validation functions
def validate_config():
    """Validate configuration settings with laptop-specific checks"""
    errors = []
    warnings = []

    # Check if CSV files exist
    if not os.path.exists(FILTERED_CSV):
        errors.append(f"Filtered CSV not found: {FILTERED_CSV}")

    if not os.path.exists(UNFILTERED_CSV):
        errors.append(f"Unfiltered CSV not found: {UNFILTERED_CSV}")

    # Check species consistency
    if len(FLOWER_SPECIES) != len(SPECIES_NAMES):
        errors.append("FLOWER_SPECIES and SPECIES_NAMES must have same length")

    # Hardware-specific checks for your laptop
    if DEVICE == "cuda":
        if not torch.cuda.is_available():
            warnings.append("CUDA not available, falling back to CPU mode")
        else:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory < 4:
                warnings.append(f"GPU has only {gpu_memory:.1f}GB memory, might be tight for large models")
            else:
                print(f"✅ RTX 3060 detected with {gpu_memory:.1f}GB VRAM - perfect for BioCLIP!")

    # Memory check for 16GB laptop
    available_ram = psutil.virtual_memory().available / 1024**3
    total_ram = psutil.virtual_memory().total / 1024**3

    print(f"💾 System RAM: {available_ram:.1f}GB available / {total_ram:.1f}GB total")

    if available_ram < 4:
        warnings.append(f"Low available RAM: {available_ram:.1f}GB - close some applications")
    elif available_ram > 8:
        print(f"✅ Plenty of RAM available ({available_ram:.1f}GB) - excellent for processing!")

    # Display results
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"   - {error}")
        return False

    if warnings:
        print("⚠️ Configuration Warnings:")
        for warning in warnings:
            print(f"   - {warning}")

    return True

# Performance monitoring functions
def get_system_stats():
    """Get current system resource usage for your laptop"""
    stats = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'available_memory_gb': psutil.virtual_memory().available / 1024**3,
        'total_memory_gb': psutil.virtual_memory().total / 1024**3
    }

    if DEVICE == "cuda" and torch.cuda.is_available():
        stats['gpu_memory_used_gb'] = torch.cuda.memory_allocated(0) / 1024**3
        stats['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        stats['gpu_memory_percent'] = (stats['gpu_memory_used_gb'] / stats['gpu_memory_total_gb']) * 100

    return stats

def print_system_stats():
    """Print current system resource usage optimized for laptop monitoring"""
    stats = get_system_stats()
    print(f"\n📊 Laptop System Resources:")
    print(f"   🔧 CPU (Ryzen 5 5600H): {stats['cpu_percent']:.1f}%")
    print(f"   💾 RAM: {stats['memory_percent']:.1f}% ({stats['available_memory_gb']:.1f}/{stats['total_memory_gb']:.1f}GB)")

    if 'gpu_memory_percent' in stats:
        print(f"   🎮 GPU (RTX 3060): {stats['gpu_memory_percent']:.1f}% ({stats['gpu_memory_used_gb']:.1f}/{stats['gpu_memory_total_gb']:.1f}GB)")

def check_thermal_throttling():
    """
    🌡️ Check if laptop might be thermal throttling (optional monitoring)
    """
    try:
        import cpuinfo
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            current_freq = cpu_freq.current
            max_freq = cpu_freq.max
            if current_freq < max_freq * 0.8:  # If running at less than 80% max frequency
                print(f"🌡️ Warning: CPU may be thermal throttling ({current_freq:.0f}MHz/{max_freq:.0f}MHz)")
            else:
                print(f"✅ CPU running at good frequency ({current_freq:.0f}MHz)")
    except:
        pass  # Skip if cpuinfo not available

# Laptop power management
def suggest_power_settings():
    """
    ⚡ Suggest optimal power settings for laptop during AI processing
    """
    print(f"\n⚡ LAPTOP POWER OPTIMIZATION TIPS:")
    print(f"   🔌 Use AC power adapter for best performance")
    print(f"   🖥️ Set Windows power plan to 'High Performance' or 'Balanced'")
    print(f"   🌡️ Ensure good ventilation to prevent thermal throttling")
    print(f"   🔇 Close unnecessary applications to free up resources")
    if DEVICE == "cuda":
        print(f"   🎮 GPU acceleration enabled - expect faster processing!")
    print(f"   📊 Monitor resource usage with print_system_stats()")
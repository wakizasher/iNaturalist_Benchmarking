# BioCleanse: A Preprocessing Pipeline for Qualitative Biodiversity Images

## Description

We present **BIOCLEANSE**, a data preprocessing pipeline which improves the curation of qualitative biodiversity data for model training and evaluation. Data from the citizen science platform iNaturalist has emerged as a key resource in AI biodiversity research, providing expert-vetted observations. This expert-vetted data denoted as Research Grade includes a date, location, image, and agreement from multiple citizen scientists on the species-level identification. 

While iNaturalist's Research Grade data is widely used in machine learning (ML) research on biodiversity, it lacks standardized image quality control, introducing significant noise into biodiversity AI workflows. BIOCLEANSE augments iNaturalist Research Grade data quality with automated data filtering for three key image quality issues:

1. **Composition** (exposure, blur)
2. **Human presence** (body part)
3. **Other species presence**

Leveraging both traditional and AI-based image quality assessment (IQA) techniques, we hypothesize curating images along these dimensions significantly improves species identification. To support rigorous assessment, we introduce several new benchmarks of different image quality and report model accuracy for zero-shot learning across three visually similar plant species.

With the release of BIOCLEANSE, we aim to catalyze progress toward real-world-ready biodiversity AI models capable of handling challenging in-situ images. We invite the biodiversity and machine learning communities to employ BIOCLEANSE on additional taxa, building more robust, generalizable AI-ready biodiversity data.

## Table of Contents

- [Installation](#installation)
- [System Requirements](#system-requirements)
- [Usage](#usage)
- [Features](#features)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)

## Installation

Install the required dependencies using pip:

`pip install -r requirements.txt`

### Key Dependencies

The project relies on several important packages:

- `torch==2.7.0` and `torchvision==0.22.0` for deep learning
- `transformers==4.37.2` for Hugging Face models
- `open-clip-torch==2.20.0` for CLIP models
- `opencv-python==4.11.0.86` for image processing
- `pandas==2.1.0` and `numpy==1.24.3` for data manipulation
- `scikit-learn==1.3.0` for machine learning utilities
- `pyiqa==0.1.13` for image quality assessment

## System Requirements

- **RAM**: At least 16GB RAM
- **GPU**: Latest NVIDIA GPU with at least 16GB VRAM (recommended)
- **Storage**: 1TB SSD for storing large image datasets
- **OS**: Compatible with Windows, Linux, and macOS

## Usage

### 1. NIQE (No-Reference Image Quality Evaluation)

Evaluate image quality using NIQE metrics:

`python NR-IQA/niqe.py`

## Configuration

### NIQE Configuration

In `niqe_analysis.ipynb`, update the CSV path:

`df = pd.read_csv(r"C:\Users\nikit\Pycharm\No_Reference\valid_niqe_results.csv")`
### VLM Configuration

In `qwen200.py`, customize the model and prompt:
```
model_id = "Qwen2.5-VL-3B-Instruct"
prompt = f"Is the image too blurry or low quality to allow identification? Answer only Yes or No."

# Update base folder path
base_folder = r"D:\iNaturalist\test_images_200"
```
### Zero-Shot Configuration

In `config.py`, set your CSV file paths:
```
FILTERED_CSV = "C:\\path\\to\\filtered_dataset.csv"
UNFILTERED_CSV = "C:\\path\\to\\unfiltered_dataset.csv"
```
## Project Structure
iNaturalist_Benchmarking/
├── Image_extraction/     # Image extraction utilities
├── NR-IQA/              # No-Reference Image Quality Assessment
├── VLM_scripts/         # Vision Language Model scripts
├── bioclip/             # BioCLIP model implementation
├── biotroveclip/        # BioTroveCLIP model implementation
├── data/                # Dataset files
├── extended_data/       # Extended benchmark data
├── misc/                # Miscellaneous utilities
├── results/             # Output results
└── tests/               # Test files
## Contributing

We welcome contributions to expand BIOCLEANSE! You can contribute by:

- **Expanding datasets**: Adding different taxa such as spiders, underwater species, birds, and other flowers
- **Improving algorithms**: Enhancing image quality assessment techniques
- **Adding benchmarks**: Creating new evaluation metrics and test cases
- **Documentation**: Improving code documentation and examples

Please feel free to submit issues, feature requests, or pull requests.

## License

This project is open source. Please refer to the LICENSE file for detailed licensing information.

## Authors

- **Nikita Gavrilov**
---

**Citation**: If you use BIOCLEANSE in your research, please consider citing our work to support continued development of biodiversity AI tools.

**Contact**: For questions, issues, or collaboration opportunities, please open an issue on this repository.

# Brain-Tumor-Segmentation-Detection-System

## Project Overview
This repository contains a comprehensive system for brain tumor segmentation and detection using deep learning models. The project includes data preprocessing, model training, evaluation, and deployment scripts, as well as utilities for handling medical imaging data.

## Folder Structure
- `appp.py`, `main.py`: Main entry points for running the application and experiments.
- `requirements.txt`: Python dependencies for the project.
- `data_mask.csv`, `user_db.json`: Data files and user database.
- `resnet-50-MRI.json`, `ResUNet-MRI.json`: Model architecture files for ResNet-50 and ResUNet.
- `utilities.py`: Utility functions for data processing and model operations.
- `app/`: Application logic and supporting modules.
  - `master.py`, `utils.py`: Core application scripts.
  - `images/`: Contains sample and processed images.
  - `pages/`: UI and workflow scripts (e.g., login, dashboard).
- `brain_env/`: Python virtual environment for dependency isolation.
- `Brain_MRI/`: Contains MRI data, trained models, and related scripts.
  - `classifier-resnet-model.json`, `classifier-resnet-weights.h5`: ResNet classifier model and weights.
  - `ResUNet-model.json`, `weights_seg.hdf5`: ResUNet segmentation model and weights.
  - `data.csv`, `data_mask.csv`: MRI data and masks.
  - `utilities.py`: MRI-specific utilities.
  - `savedmodel/`: Saved model files for deployment.
  - `TCGA_*`: Folders containing MRI scan data for different patients.

## Setup Instructions
1. **Clone the repository:**
   ```powershell
   git clone https://github.com/Harsh456B/Brain-Tumor-Segmentation-Detection-System.git
   cd Brain-Tumor-Segmentation-Detection-System
   ```
2. **Create and activate the Python environment:**
   ```powershell
   python -m venv brain_env
   .\brain_env\Scripts\Activate.ps1
   ```
3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Run the application:**
   ```powershell
   python main.py
   ```

## Preprocessing Steps
- Data is loaded from `Brain_MRI/data.csv` and masked using `data_mask.csv`.
- Images are normalized and resized for model input.
- Data augmentation is performed for robust training.

## Model Training & Evaluation
- ResNet-50 and ResUNet architectures are defined in their respective JSON files.
- Training scripts are provided in `main.py` and `app/master.py`.
- Model weights are saved in `.h5` and `.hdf5` formats for later use.

## Deployment
- Trained models can be loaded for inference using the provided scripts.
- The application supports user authentication and result visualization.

## Utilities
- Utility scripts for image processing, data handling, and evaluation are available in `utilities.py` files.

## Notes
- MRI data folders (`TCGA_*`) are large and may be excluded from pushes if needed.
- Ensure you have sufficient disk space and memory for training and inference.



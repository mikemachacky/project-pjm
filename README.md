# PJM Sign Language Recognition - Project Overview

## Description
This project implements gesture classification for Polish Sign Language (PJM) using models based on BiLSTM with attention and 1D CNN architectures. The input consists of frame-by-frame hand tracking features extracted from sign language videos and real-time recordings. The models are trained on custom datasets and saved in the `model/` directory.

## Project Structure
```
project-pjm/
├── archive/                  # Old notebooks and experiment results
├── augmentation/             # Notebooks related to data augmentation
├── dataset/                  # Raw dataset files and tools for recording gestures
├── model/                    # Trained models and labels
├── real-time/                # Inference and testing notebooks
├── PJM-sign-language.db      # Main SQLite database
├── requirements.txt          # List of required Python packages
```

## How to Run the Project
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/project-pjm.git
   cd project-pjm
   ```

2. **Set up a Python virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Explore the notebooks**
   Start with:
   - `real-time/2025-07-14.ipynb` – for model evaluation and inference
   - `augmentation/tf_dataset.ipynb` – for preprocessing and augmentation

4. **Model Weights and Labels**
   Trained models are stored in `model/YYYY-MM-DD/`:
   - `bilstm_attention_model.keras`
   - `cnn_1d_model.keras`
   - `labels.txt` – contains the label encoder mapping used for classification

   Example usage to load a model:
   ```python
   from tensorflow.keras.models import load_model
   import joblib

   model = load_model("model/2025-07-14/bilstm_attention_model.keras")
   labels = joblib.load("model/2025-07-14/labels.txt")
   ```

5. **Real-Time Prediction (Optional)**
   Real-time prediction scripts can be found in the `real-time/` directory.

## Requirements
The list of required packages is available in `requirements.txt`.

---
Supervisor can explore any of the provided `.ipynb` notebooks for detailed insight into model training, evaluation metrics, and saved models for static and dynamic gestures.

---
Author: [Magdalena Machacka]
Date: 2025-07-14

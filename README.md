# Leveraging Convolutional Neural Network for Classification of Leaf Disease

## Overview
This project implements a deep learning approach to detect and classify plant leaf diseases into three categories: **Healthy**, **Powdery**, and **Rust**. Leveraging a custom CNN architecture with image preprocessing and data augmentation, the model aims to provide an accessible, accurate, and scalable solution for early plant disease detection to improve crop management and food security.

---

## Features
- **Automatic Disease Identification** using CNN
- **Three-Class Classification**: Healthy, Powdery, Rust
- **Data Augmentation**: Random shear, zoom, horizontal flipping
- **Model Optimization**: Tuned epochs and neuron counts for best performance
- **Evaluation Metrics**: Precision, Recall, F1-Score, Confusion Matrix
- **Comparative Analysis**: Benchmark against Naïve Bayes and MLP models

---

## Dataset
- **Total Images**: 1,532 leaf images
- **Categories**:
  - Healthy: 458 images
  - Powdery: 430 images
  - Rust: 434 images
- **Splits**:
  - Training Set: Majority of the images
  - Validation Set: 60 images
  - Test Set: 150 images

All images were resized to `225 x 225 pixels` and normalized to `[0,1]`.

---

## Methodology

1. **Data Acquisition**
   - Collected high-resolution images under varied conditions.
2. **Preprocessing**
   - Normalization, resizing, augmentation, label encoding.
3. **Model Design**
   - CNN with multiple convolutional and pooling layers.
   - Dense layers with softmax activation for classification.
4. **Training**
   - Optimizer: Adam
   - Loss Function: Categorical Cross-Entropy
   - Optimal configuration: **10 epochs, 64 neurons**
5. **Evaluation**
   - Tracked accuracy, precision, recall, F1-score across classes.

---

## Results
| Class    | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Healthy  | 0.92      | 0.88   | 0.90     |
| Powdery  | 0.85      | 0.90   | 0.87     |
| Rust     | 0.89      | 0.84   | 0.86     |

- **Overall Accuracy**: ~90%
- **Best Performance**: Healthy leaf classification
- **Confusion Matrix**: Shows minor confusion between Powdery and Rust categories

---

## Comparative Analysis
CNN outperformed traditional classifiers:

| Model        | Avg Precision | Avg Recall | Avg F1-Score |
|--------------|---------------|------------|--------------|
| Naïve Bayes  | ~0.75         | ~0.75      | ~0.66        |
| MLP          | ~0.82         | ~0.77      | ~0.74        |
| **CNN**      | **0.89**      | **0.87**   | **0.87**     |

---

## Tech Stack
- **Language**: Python 3.10
- **Libraries**:
  - TensorFlow
  - Keras
  - NumPy
  - OpenCV
- **Platform**: Linux
- **Hardware**: NVIDIA GPUs for acceleration

---

## Future Work
- Expand dataset with more samples and diverse conditions.
- Experiment with advanced CNN architectures (e.g., EfficientNet).
- Deploy the model in real farm settings.
- Integrate a mobile/web interface for farmers.

---

## Authors
- Amogh Gadad
- Amruta Biradarpatil
- Ashtami Hosapeti
- Naman Timmapur

Department of Electronics and Communication Engineering,  
KLE Technological University, Dr. M.S. Sheshgiri Campus, Belagavi, India

# Pinpoint AI 
PinpointAI is a machine learning project that uses computer vision to predict a location anywhere in France from a single Google Street View image. By leveraging state-of-the-art deep learning models and geospatial data, this project aims to identify GPS coordinates from a Streetview image with remarkable accuracy.


---
# 🚧 WIP 
## Features

- **Data Preprocessing and Augmentation**: Ensures robust image recognition by preparing diverse and scalable datasets.
- **Deep Learning Pipeline**: Utilizes transfer learning with pre-trained models like ResNet and EfficientNet.
- **Location Prediction**:
  - **Classification**: Predicts regions or countries.
  - **Regression**: Estimates GPS coordinates.
- **Evaluation Metrics**: Includes geodesic distance calculations and accuracy scoring.

---

## Steps Overview

### 1. Collect Data
- **Data Source**: Use Google Street View API or other similar datasets to gather labeled images of various locations worldwide.
- **Labels**: Each image must be labeled with its GPS coordinates.
- **Data Diversity**: Data includes diverse environments (urban, rural, mountainous, coastal, etc.) and conditions (weather, time of day).

### 2. Preprocessing
- **Image Preprocessing**: Resize, normalize, and augment images (rotation, cropping, brightness adjustment) to ensure robustness.

### 3. Model Design
- **Base Model**: Used a pre-trained Convolutional Neural Network (CNN): ResNet feature extraction.
- **Custom Layers**: Add fully connected layers to map features to location predictions.
- **Loss Function**:
  - Regression (predicting GPS coordinates): Use mean squared error or a custom geospatial distance loss.

### 4. Training
- **Transfer Learning**: Fine-tune a pre-trained model on your dataset to save training time and improve performance.
- **Regularization**: Use dropout and data augmentation to prevent overfitting.
- **Hardware**: A high-end GPU or TPU is necessary for training on large-scale image datasets.

### 5. Challenges
- **Geographical Bias**: Some locations may have distinct features, while others are generic (e.g., highways).
- **Scalability**: Training on a global scale requires significant computational resources.
- **Legal Concerns**: Ensure compliance with Google's terms of service if using their data.

### 6. Evaluation
- Use test data from regions not included in the training set to evaluate the model's generalization capability.
- **Metrics**:
  - For classification: Accuracy, F1 score, etc.
  - For regression: Mean geodesic distance error.

### 7. Potential Improvements
- **Combine Modalities**: Use metadata (e.g., compass direction, time of year) alongside the image for improved accuracy.
- **Hierarchical Modeling**: First predict a broad region, then refine the location prediction within it.
- **Crowdsourced Feedback**: Use user inputs to refine predictions and retrain the model.

---

## Getting Started

### Prerequisites
- Python 3.8+
- Libraries:
  - `torch`
  - `torchvision`
  - `numpy`
  - `Pillow`
  - `geopy`
  - `scikit-learn`
- Google Maps API key for data collection

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/PinpointAI.git
   cd PinpointAI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Google Maps API key in a `.env` file:
   ```env
   GOOGLE_MAPS_API_KEY=your_api_key_here
   ```

---

## Usage

### 1. Data Collection
Run the `data_collection.py` script to fetch images from Google Street View:
```bash
python data_collection.py
```

### 2. Model Training
Train the model using the `train.py` script:
```bash
python train.py
```

### 3. Evaluation
Evaluate the trained model with test data:
```bash
python evaluate.py
```

---

## Project Structure
```
PinpointAI/
├── data/               # Raw and processed datasets
├── models/             # Pre-trained and trained models
├── scripts/            # Scripts for training, evaluation, and data collection
├── utils/              # Utility functions (preprocessing, geodesic metrics, etc.)
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
```

[Back to top 🚀](#top)


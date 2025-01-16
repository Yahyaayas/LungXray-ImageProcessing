# Chest X-Ray Segmentation using U-Net

## Our Team
Danish Dhiaurrahman Ritonga (22/492936/TK/53966)	
Muhamad Bintang Harry Dharmawan (22/502957/TK/54927)
Yahya Ayas Firdausi (22/503609/TK/55039)
Ariq Fadhilah Jeha (22/504174/TK/55127)

## Project Overview
This project focuses on the segmentation of lung regions from chest X-ray images using the U-Net architecture. It aims to assist medical professionals by automating the analysis of chest X-ray images, potentially improving diagnostic accuracy and reducing workload.

## Motivation
Medical imaging plays a critical role in diagnosis and treatment. However, the manual analysis of these images is time-consuming and prone to errors. By leveraging deep learning, specifically U-Net, we aim to:

- Identify diseases through lung segmentation.
- Automate the monitoring of patient conditions.
- Simplify the analysis process for healthcare providers.

## Methodology
### 1. **Data Preparation**
   - **Load Data**: Images and masks are loaded from a specified directory.
   - **Data Augmentation**: Augment data to make the model robust to variations.
   - **Train-Test Split**: Split the dataset into training and testing subsets.

### 2. **U-Net Architecture**
   - U-Net comprises an encoder for feature extraction and a decoder for segmentation mask generation.

### 3. **Model Training**
   - The model is trained for 50 epochs using TensorFlow, with data generators to handle large datasets efficiently.

### 4. **Evaluation Metrics**
   - **Intersection over Union (IoU)**: Measures model performance by comparing predicted segmentation with ground truth.

### 5. **Image Post-Processing**
   - **Otsu Thresholding**: Converts predicted masks to binary.
   - **Contour Detection**: Extracts the edges of segmented lungs to remove irrelevant regions.

## Results
- IoU scores achieved:
  - Example 1: 0.94
  - Example 2: 0.89
  - Example 3: 0.96
  - Example 4: 0.94

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- OpenCV
- Google Colab (optional, for running the provided notebook)

## Getting Started
1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook Model_Training_Lung_Segmentation_using_U_Net.ipynb
   ```

## File Descriptions
- `Model_Training_Lung_Segmentation_using_U_Net.ipynb`: Main notebook for training and evaluating the U-Net model.
- `data/`: Directory containing chest X-ray images and their corresponding masks.
- `results/`: Contains predicted segmentation outputs and evaluation results.

## Acknowledgments
- **Team Members**:
  - Danish Dhiaurrahman Ritonga
  - Muhamad Bintang Harry Dharmawan
  - Yahya Ayas Firdausi
  - Ariq Fadhilah Jeha
- Special thanks to UGM for their support.

## References
- [U-Net Architecture](https://arxiv.org/abs/1505.04597)
- [TensorFlow Documentation](https://www.tensorflow.org/)

---
"A good key can open any door, but the keyholder must wisely choose the right one."

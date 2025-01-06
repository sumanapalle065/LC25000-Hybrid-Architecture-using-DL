[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sumanapalle065/LC25000-Hybrid-Architecture-using-DL/blob/main/LC25000_Hybrid_Arc.ipynb)

# LC25000 Hybrid Architecture using DL
This project explores the LC25000 Lung and Colon Histopathological Dataset.
The goal of the project is to develop a Deep Learning Model architecture to demonstrate an effectiveness of a proposed approach in clinical settings for early 
cancer detection, leading to improved treatment outcomes and reduced mortality rates.

# Tri-Fusion Model for Lung and Colon Cancer Classification

This repository contains the implementation and analysis of the **Tri-Fusion Model**, a deep learning-based framework for classifying histopathological images of lung and colon cancers. The model leverages advanced preprocessing techniques and cutting-edge neural networks to achieve high classification accuracy and reliable predictions. The TriFusionNet model combines three pre-trained architectures—MobileNetV2, InceptionV3, and EfficientNetB7—leveraging their strengths for enhanced lung and colon cancer classification. These models, pre-trained on ImageNet weights, are fine-tuned with custom convolutional and dense layers to maximize feature extraction, reduce dimensionality, and achieve robust multi-class classification.

---

## **Performance Metrics**
The Tri-Fusion Model has been evaluated using multiple performance metrics to provide a comprehensive understanding of its effectiveness:

- **Accuracy:** 99.67%
- **Loss:** 0.0114
- **Precision:** 0.9975
- **F1-Score:** 0.9975
- **Sensitivity (Recall):** 1.0
- **Specificity:** 1.0

These results demonstrate the robustness and reliability of the model for lung and colon cancer classification tasks.

---

## **Dataset Description**

The model was trained and tested on the **LC25000 Dataset**, which includes 25,000 histopathology images of lung and colon cancer. The dataset comprises:
- **Lung Cancer Types:**
  - Adenocarcinoma (ACAL)
  - Squamous Cell Carcinoma (SCCL)
  - Benign Tissue of Lung (BTL)
- **Colon Cancer Types:**
  - Adenocarcinoma of Colon (ACAC)
  - Benign Tissue of Colon (BTC)

The dataset is balanced, with 5,000 images per class, ensuring consistent representation across all cancer types. Images were preprocessed to a size of **148 x 148 pixels** for analysis.

---

## **Preprocessing Steps**

The preprocessing pipeline includes several key steps to enhance image quality and prepare the dataset for classification:
1. **Scaling:** Images resized to 148 x 148 pixels for uniformity.
2. **Gaussian Blur:** Reduces noise and smooths the image for better clarity.
3. **Denoising:** Removes visible noise without distorting key features.
4. **Segmentation:** Extracts regions of interest to focus on relevant tissue features.
5. **Normalization:** Ensures pixel values are scaled consistently across all images.

### Sample Preprocessing Stages
1. **Original Image**
2. **Gaussian Blur Applied**
3. **Denoised Image**
4. **Segmented Image**

---

## **Model Architecture**

The **Tri-Fusion Model** combines the strengths of multiple neural networks for robust feature extraction and classification. The architecture includes:
- **Preprocessing pipeline** for input image preparation.
- **Fusion of three feature extractors** for enhanced classification performance.

---

## **Experimental Results**

### **Key Findings:**
- The model achieved an **accuracy of 99.67%** and an **F1-Score of 0.9975**, showcasing its reliability.
- The loss values stabilized at **0.004 (training)** and **0.0114 (testing)**, indicating effective learning without overfitting.

### **Graphical Representations**
1. **Accuracy Graph:** Demonstrates the steady increase in training accuracy, with minor fluctuations in testing accuracy.
2. **Loss Graph:** Shows a decreasing trend in training loss and eventual stabilization of testing loss.
3. **ROC Curve:** All classes achieved an **AUC of 1.0**, indicating flawless classification.
4. **Precision-Recall Curve:** AP scores range between **0.99 to 1.0** across all classes.

### **Performance Summary**
| Metric      | Value   |
|-------------|---------|
| **Accuracy** | 99.67%  |
| **Loss**     | 0.0114  |
| **Precision**| 0.9975  |
| **F1-Score** | 0.9975  |
| **Sensitivity** | 1.0 |
| **Specificity** | 1.0 |

---

## **Model Visualizations**
The repository includes visualizations of predictions and preprocessing steps:
1. **Predicted Images:** Displays classifications of histopathological samples into distinct classes, highlighting morphological differences.
2. **Feature Extraction:** Demonstrates how the model identifies unique patterns in tissue samples for accurate classification.

---

## **How to Use**

### **Prerequisites**
- Python 3.8+
- TensorFlow/Keras
- NumPy, Pandas, Matplotlib, and other standard libraries




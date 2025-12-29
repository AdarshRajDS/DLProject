# 🦠 Mold vs No-Mold Image Classification (CNN – PyTorch)

This project implements a Convolutional Neural Network (CNN) to classify images as **Mold** or **No-Mold**.  
The dataset contains annotated mold images and manually collected non-mold images, organized as:


The project includes:
- Dataset preparation & train/val/test splitting  
- Custom PyTorch Dataset & DataLoaders  
- Lightweight CNN model  
- Complete training loop with accuracy/loss curves  
- Evaluation metrics  
- Model saving & loading  
- Kaggle-ready inference script  

---

## 📁 Dataset

### **Structure**
The dataset is divided into two folders:

- `mold/` — images containing visible mold
- `nomold/` — clean surfaces with **no mold**
- `Dataset` - https://drive.google.com/file/d/1Vm1Ff9qb1JCLTmkXXwV9yuGm7G27540i/view?usp=sharing

Images are resized and augmented using:
- Random horizontal flip  
- Random rotation  
- Color jitter  
- Normalization  

### **Why we need No-Mold images**
The problem is **binary classification**.  
If your dataset contains only mold images, the model cannot learn the difference between *mold* and *not mold*.

Therefore both classes are required.

---

## 🧠 Model

A lightweight CNN was built using:
- 3 convolution layers  
- BatchNorm + ReLU  
- MaxPooling  
- Fully connected classifier  
- Dropout for regularization  

This keeps the model:
- Small  
- Fast  
- Suitable for small datasets

---

## 🚀 Training

The training script includes:
- Train/val split (70/15/15)
- Balanced batch sampling
- Adam optimizer
- Cross-entropy loss
- Learning rate scheduling
- Early stopping
- Model checkpoint saving (`mold_cnn_best.pth`)



# 🦠 Mold vs No-Mold Image Classification (CNN – PyTorch)

This project implements a **Convolutional Neural Network (CNN)** to classify images as **Mold** or **No-Mold**.  
The dataset contains annotated mold images and manually collected non-mold images for binary classification.

The project includes:
- **Automated dataset preparation** with balanced train/val/test splitting  
- **Custom PyTorch DataLoaders** with image augmentation  
- **Lightweight CNN model** (~1.6M parameters)  
- **Complete training pipeline** with validation monitoring  
- **Evaluation metrics** (accuracy, confusion matrix, classification report)  
- **Model checkpointing** (saves best model automatically)  
- **Kaggle-ready inference function** for predictions  

---

## 📁 Project Structure

```
DLProject/
├── input_root/              # Original dataset
│   ├── mold/               # Images with mold
│   └── nomold/             # Images without mold
├── output_root/            # Processed dataset (auto-generated)
│   ├── train/              # Training set (70%)
│   ├── val/                # Validation set (15%)
│   └── test/               # Test set (15%)
├── models/                 # Saved model checkpoints
│   └── mold_cnn_best.pth  # Best model weights
├── working/                # Training outputs (auto-generated)
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── mold_cnn_artifacts.zip
├── notebook254bbc3ecc.ipynb  # Main notebook
└── README.md
```

---

## 🚀 Getting Started

### **Prerequisites**

Install the required packages:

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib pillow
```

Or if you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

### **Dataset Preparation**

1. **Place your raw images** in the following structure:
   ```
   input_root/
   ├── mold/       # All mold images
   └── nomold/     # All no-mold images
   ```

2. The notebook will **automatically**:
   - Balance the dataset (equal samples from both classes)
   - Split into train (70%), validation (15%), and test (15%)
   - Create the `output_root/` directory structure
   - Copy files to appropriate folders

---

## 📝 How to Run

### **Option 1: Run the Complete Notebook**

Open `notebook254bbc3ecc.ipynb` and execute all cells in order:

1. **Cell 1**: Environment setup and explore input directory
2. **Cell 2**: Dataset balancing and train/val/test splitting
3. **Cell 3**: Create PyTorch DataLoaders with augmentation
4. **Cell 4**: Complete training pipeline (includes all steps below)

### **Option 2: Run Individual Steps**

The notebook is modular. You can run:

#### **Step 1: List Input Files**
```python
# Cell 1 - Explore the dataset
import os
for dirname, _, filenames in os.walk('./input_root'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

#### **Step 2: Split Dataset**
```python
# Cell 2 - Creates balanced train/val/test splits
# Automatically balances classes and copies files to output_root/
```

#### **Step 3: Create DataLoaders**
```python
# Cell 3 - Sets up PyTorch DataLoaders with augmentation
# Includes resizing, flipping, rotation, normalization
```

#### **Step 4: Train Model**
```python
# Cell 4 - Complete training pipeline
# - Defines SmallCNN architecture
# - Trains for 25 epochs with validation
# - Saves best model based on validation accuracy
# - Generates training curves
# - Evaluates on test set
# - Creates confusion matrix
# - Provides inference function
```

---

## 🧠 Model Architecture

**SmallCNN** - Lightweight CNN optimized for small datasets:

```
- Conv2D(3→32) + ReLU + BatchNorm → MaxPool(2×2)
- Conv2D(32→64) + ReLU + BatchNorm → MaxPool(2×2)
- Conv2D(64→128) + ReLU + BatchNorm → MaxPool(2×2)
- Conv2D(128→256) + ReLU + BatchNorm → AdaptiveAvgPool
- Flatten → Dropout(0.4) → Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→2)
```

**Key features:**
- Only ~1.6M parameters (very lightweight)
- Batch normalization for stability
- Dropout layers to prevent overfitting
- Adaptive pooling for flexible input sizes

---

## 🎓 Training Configuration

| Parameter | Value |
|-----------|-------|
| **Image Size** | 224×224 |
| **Batch Size** | 16 |
| **Epochs** | 25 |
| **Optimizer** | Adam (lr=1e-4) |
| **Loss Function** | CrossEntropyLoss |
| **Scheduler** | ReduceLROnPlateau (monitors val_acc) |
| **Device** | CUDA (if available) else CPU |

### **Data Augmentation (Training only)**
- Resize to 224×224
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Normalization (ImageNet stats)

### **Validation & Test**
- Resize to 224×224
- Normalization only (no augmentation)

---

## 📊 Training Output

After training completes, you'll find:

### **1. Model Checkpoint**
- **Location**: `models/mold_cnn_best.pth`
- Contains: model weights, optimizer state, class names
- Saved automatically when validation accuracy improves

### **2. Training Curves**
- **Location**: `working/training_curves.png`
- Shows: train/val loss and accuracy over epochs

### **3. Confusion Matrix**
- **Location**: `working/confusion_matrix.png`
- Shows: classification performance on test set

### **4. Console Output**
```
Epoch 1/25 — train_loss: 0.6234, train_acc: 0.6521, val_loss: 0.5432, val_acc: 0.7234, time: 12.3s
Saved new best model (val_acc=0.7234) -> ./models/mold_cnn_best.pth
...
Test accuracy: 0.8542
Classification report:
              precision    recall  f1-score   support
        mold       0.87      0.84      0.85       123
      nomold       0.84      0.87      0.86       127
```

---

## 🔮 Making Predictions

### **Using the Inference Function**

The notebook includes a ready-to-use `predict_image()` function:

```python
from PIL import Image

def predict_image(image_path, model, class_names, device, resize=224):
    """
    Predicts class for a single image
    
    Args:
        image_path: Path to image file
        model: Trained PyTorch model
        class_names: List of class names ['mold', 'nomold']
        device: torch.device
        resize: Image size (default 224)
    
    Returns:
        Dictionary with prediction index, label, and probabilities
    """
    # See Cell 4 for full implementation
```

### **Example Usage**

```python
# Load the saved model
checkpoint = torch.load('./models/mold_cnn_best.pth')
model.load_state_dict(checkpoint['model_state'])
class_names = checkpoint['class_names']

# Predict on new image
result = predict_image('path/to/image.jpg', model, class_names)
print(f"Prediction: {result['pred_label']}")
print(f"Confidence: {result['probs'][result['pred_index']]:.2%}")
```

### **Output Format**

```python
{
    'pred_index': 0,
    'pred_label': 'mold',
    'probs': array([0.92, 0.08])  # [mold_prob, nomold_prob]
}
```

---

## 📈 Expected Results

With proper training, you should achieve:

- **Training Accuracy**: ~85-95% (depends on dataset quality)
- **Validation Accuracy**: ~80-90%
- **Test Accuracy**: ~80-90%

**Note**: Results vary based on:
- Dataset size and quality
- Image diversity
- Class balance
- Augmentation strength

---

## 🛠️ Customization

### **Change Hyperparameters**

Edit these variables in Cell 4:

```python
batch_size = 16          # Increase for faster training (if GPU memory allows)
image_size = 224         # Larger = more detail but slower
num_epochs = 25          # More epochs = longer training
seed = 42                # For reproducibility
```

### **Modify Data Augmentation**

Edit the `data_transforms` dictionary in Cell 3 or Cell 4:

```python
transforms.RandomRotation(15),           # Change rotation angle
transforms.ColorJitter(brightness=0.15), # Adjust brightness range
```

### **Change Model Architecture**

Modify the `SmallCNN` class in Cell 4 to add/remove layers or change channel sizes.

---

## ⚠️ Troubleshooting

### **"CUDA out of memory"**
- Reduce `batch_size` to 8 or 4
- Reduce `image_size` to 128 or 160

### **"Dataset not found"**
- Ensure `input_root/` contains `mold/` and `nomold/` folders
- Check that folders contain actual image files

### **Low accuracy**
- Increase dataset size (collect more images)
- Train for more epochs
- Increase data augmentation
- Try transfer learning (ResNet, EfficientNet)

### **Overfitting (train acc >> val acc)**
- Increase dropout rates
- Add more data augmentation
- Collect more training data
- Reduce model complexity

---

## 📦 Deliverables

This project generates:

1. ✅ Trained model: `models/mold_cnn_best.pth`
2. ✅ Training curves: `working/training_curves.png`
3. ✅ Confusion matrix: `working/confusion_matrix.png`
4. ✅ Artifacts archive: `working/mold_cnn_artifacts.zip`
5. ✅ Inference function for production use

---

## 🎯 Next Steps

To improve the model further:

1. **Collect more data** - Especially edge cases
2. **Try transfer learning** - Use pre-trained ResNet or EfficientNet
3. **Experiment with augmentation** - Test different strategies
4. **Ensemble models** - Combine multiple models
5. **Deploy** - Create a web app or API endpoint

---

## 📄 License

This project is for educational purposes. Please ensure you have proper rights to use the dataset images.

---

## 👥 Contributors

- Deep Learning Project Team
- Dataset: Roboflow + Custom Collection

---

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Roboflow for dataset annotation tools
- Kaggle for compute resources



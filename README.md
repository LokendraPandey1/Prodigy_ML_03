# Cat vs Dog Image Classification — Support Vector Machine

**Task 03 — Machine Learning Internship @ Prodigy InfoTech**

An SVM model that classifies images of cats and dogs using HOG (Histogram of Oriented Gradients) feature extraction.

## Dataset

Kaggle's "Dogs vs Cats" dataset:

```
Images/
    Cat/   (12,501 images)
    Dog/   (12,501 images)
```

Full dataset — 25,002 images total.

## Approach

SVMs don't work directly on raw image pixels for this kind of task — pixel values alone don't capture shape/texture information well enough to be linearly (or even kernel-) separable. So instead of feeding raw pixels to the SVM, this pipeline extracts **HOG features**, which capture edge/gradient structure (useful for distinguishing shapes like ear/face contours), then reduces dimensionality with PCA before classification.

## Files

- `svm_cat_dog_classifier.py` — full pipeline: image loading, HOG feature extraction, PCA, SVM training with grid search, evaluation, and model export
- `confusion_matrix.png` — confusion matrix on the held-out test set
- `svm_cat_dog_model.joblib` — saved pipeline (SVM + scaler + PCA) for classifying new images

## How It Works

1. **Load images** — auto-detects `Cat/`/`Dog/` subfolders (case-insensitive) or filenames containing "cat"/"dog"
2. **Subsample** — 1,500 images per class used for training (3,000 total) to keep SVM training time tractable; classical SVMs scale poorly with dataset size and HOG dimensionality
3. **Feature extraction** — each image is:
   - Converted to grayscale
   - Resized to 64×64
   - Passed through HOG (9 orientations, 8×8 pixel cells, 2×2 cell blocks) → 1,764-dim feature vector
4. **Scaling** — `StandardScaler` applied to features
5. **Dimensionality reduction** — PCA to 150 components (retains ~67% of variance), which significantly speeds up SVM training
6. **Model training** — `GridSearchCV` over an RBF-kernel SVM, tuning `C` and `gamma`, with 3-fold cross-validation and balanced class weights
7. **Evaluation** — accuracy, precision/recall/F1, and a confusion matrix on a 20% held-out test split
8. **Export** — the fitted SVM, scaler, and PCA are bundled into a single `.joblib` file, along with a `predict_image()` helper to classify new images

## Results

| Metric | Value |
|---|---|
| Best hyperparameters | C=10, gamma=0.001, kernel='rbf' |
| Cross-validation accuracy | 74.53% |
| **Test accuracy** | **77.0%** |

**Classification report:**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Cat | 0.78 | 0.75 | 0.77 |
| Dog | 0.76 | 0.79 | 0.77 |

## Limitations & Next Steps

- 77% is a reasonable ceiling for classical HOG+SVM on this dataset — cat/dog images have overlapping textures, poses, and backgrounds that are hard to separate without learned (CNN) features. Deep learning approaches (e.g., a CNN or transfer learning with a pretrained model like ResNet/MobileNet) typically reach 90%+ on the same task.
- Only 3,000 of the 25,002 available images were used for training (1,500 per class), purely for runtime reasons — training on more of the full set (with a more efficient kernel or a linear SVM on more PCA components) could improve results further.
- Trying alternate features (color histograms, LBP texture features) or combining HOG with color-based features could improve accuracy without moving to deep learning.

## Requirements

```
numpy
opencv-python
scikit-image
scikit-learn
matplotlib
joblib
```

## Usage

Train the model:
```bash
python3 svm_cat_dog_classifier.py
```

Classify a new image after training:
```python
from svm_cat_dog_classifier import predict_image
result = predict_image("path/to/new_image.jpg")
print(result)  # "Cat" or "Dog"
```

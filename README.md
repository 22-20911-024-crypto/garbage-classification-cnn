# \u267b\ufe0f Garbage Classification using CNN & Transfer Learning

A Deep Learning capstone project that classifies waste images into 6 categories \u2014 **cardboard, glass, metal, paper, plastic, trash** \u2014 using a CNN trained from scratch and compared against transfer-learning models (MobileNetV2, ResNet50).

> Built as part of an Artificial Intelligence course capstone project.

\ud83d\udd17 **Live Demo:** [Streamlit App](#) *(add your Streamlit Cloud link here)*

---

## \ud83d\udccc Problem Statement

Improper waste sorting is a major bottleneck in global recycling systems. Automated visual waste classification can power smart recycling bins, robotic sorting arms, and consumer recycling apps \u2014 improving sorting accuracy at scale.

**Formal statement:** Given an image of a single waste item, classify it into one of six categories: cardboard, glass, metal, paper, plastic, or trash.

## \ud83d\udcca Dataset

- **Source:** Garbage Classification Dataset (TrashNet-based), via Kaggle
- **Size:** 2,527 images across 6 classes (moderately imbalanced \u2014 trash: 137 vs paper: 594)
- **Why this dataset:** A widely recognized benchmark in waste-classification research, with clean, consistently-sized images suitable for a full CNN + transfer-learning pipeline.

## \ud83d\udee0\ufe0f Approach

1. **EDA** \u2014 class distribution, sample images, image size/quality checks
2. **Preprocessing** \u2014 resize to 224x224, augmentation (train only), stratified 70/15/15 split
3. **Class imbalance handling** \u2014 class weights + augmentation for the minority class
4. **3 Models** \u2014 Baseline CNN (from scratch), MobileNetV2 (transfer learning), ResNet50 (transfer learning)
5. **Evaluation** \u2014 Accuracy, Precision, Recall, F1-score, confusion matrix
6. **Explainability** \u2014 Grad-CAM visualizations
7. **Deployment** \u2014 Streamlit web app for live image classification

## \ud83d\udcc8 Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **ResNet50** \u2b50 | **83.2%** | **80.6%** | **83.2%** | **80.9%** |
| MobileNetV2 | 75.5% | 72.9% | 76.7% | 73.6% |
| Baseline CNN | 67.6% | 67.3% | 68.9% | 65.5% |

**Best Model:** ResNet50 (Transfer Learning) \u2014 chosen for highest test F1-score. Transfer learning substantially outperformed the from-scratch baseline, showing the value of ImageNet-pretrained features for this task.

## \ud83d\uddc2\ufe0f Project Structure

```
\u251c\u2500\u2500 app.py                  # Streamlit web application
\u251c\u2500\u2500 best_model.keras        # Trained ResNet50 model (tracked via Git LFS)
\u251c\u2500\u2500 class_indices.json      # Class label mapping
\u251c\u2500\u2500 requirements.txt
\u251c\u2500\u2500 *.ipynb                 # Full notebook: EDA \u2192 Preprocessing \u2192 CNN \u2192 Transfer Learning \u2192 Grad-CAM
\u2514\u2500\u2500 README.md
```

## \ud83d\ude80 Run Locally

```bash
git clone https://github.com/22-20911-024-crypto/garbage-classification-cnn.git
cd garbage-classification-cnn
pip install -r requirements.txt
streamlit run app.py
```

## \ud83e\uddf0 Tools & Technologies

Python \u00b7 TensorFlow/Keras \u00b7 MobileNetV2 \u00b7 ResNet50 \u00b7 OpenCV \u00b7 Scikit-learn \u00b7 Matplotlib \u00b7 Seaborn \u00b7 Streamlit \u00b7 Google Colab (GPU)

## \ud83d\udc64 Author

**Laraib Nadeem**
Artificial Intelligence \u2014 Machine Learning & Deep Learning

# Heart Disease Prediction using Naive Bayes

## Overview
This project aims to predict the presence of heart disease in patients using the Naive Bayes algorithm. The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).

## Dataset
- **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Description:** The dataset consists of 303 instances and 14 attributes, including features such as age, sex, chest pain type, resting blood pressure, cholesterol levels, and more.
- **Target:** The "target" variable indicates the presence (1) or absence (0) of heart disease.

## Prerequisites
Install the following Python packages:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Project Structure
- `dataset.csv` - Heart disease dataset
- `heart_disease_prediction.py` - Main script for training and evaluating the model

## Steps
1. **Data Loading:**
```python
import pandas as pd
dataset = pd.read_csv('/content/dataset.csv')
X = dataset.iloc[:, :13].values
y = dataset['target'].values
```

2. **Train-Test Split:**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=90)
```

3. **Feature Scaling:**
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

4. **Model Training:**
```python
from sklearn.naive_bayes import GaussianNB
nvclassifier = GaussianNB()
nvclassifier.fit(X_train, y_train)
```

5. **Prediction and Evaluation:**
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Predictions
y_pred = nvclassifier.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_plot = ConfusionMatrixDisplay(cm, display_labels=[1, 0])
cm_plot.plot()
plt.show()

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
```

## Results
Sample output metrics:
```
Accuracy: 0.85
Precision: 0.83
Recall: 0.88
F1-score: 0.85
```

## Conclusion
- The Naive Bayes algorithm shows promising performance in predicting heart disease.
- Further improvements can be made by tuning hyperparameters or trying other algorithms.

## Author
- **Khushi Jain**

## Acknowledgments
- UCI Machine Learning Repository
- Scikit-Learn Library


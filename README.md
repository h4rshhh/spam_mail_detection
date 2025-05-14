
# 🧠 Logistic Regression for Imbalanced Binary Classification

This project tackles the challenge of **imbalanced binary classification** using **Logistic Regression**, with techniques such as **class weighting** and **SMOTE (Synthetic Minority Over-sampling Technique)** to boost performance, especially for the minority class.

## Problem Statement

In many real-world scenarios, such as fraud detection or medical diagnosis, one class is underrepresented (minority), which leads to biased models. This project demonstrates how to deal with such class imbalance and build a model that performs well across both classes.

---

## Dataset Overview

The dataset consists of features (`X`) and binary labels (`Y`), where:

- Class `0` is the **majority class**.
- Class `1` is the **minority class**.

---

##  Workflow

### 1. Baseline Model (Unbalanced)

- Trained a `LogisticRegression` model without handling class imbalance.
- Observed **high accuracy** but **poor recall** for the minority class (`1`).
- **Conclusion**: Model favors majority class — not suitable for imbalanced data.

###  2. Balanced Logistic Regression (class_weight='balanced')

- Used `LogisticRegression(class_weight='balanced')`.
- Performance improved significantly on class `1`.
- Accuracy, precision, recall, and F1-score were much more balanced.

###  3. SMOTE (Synthetic Minority Over-sampling Technique)

- Applied SMOTE on the training data to balance class distribution.
- Trained another logistic regression model on the SMOTE-generated dataset.
- Achieved **even better** performance with high precision and recall for class `1`.

---

## Evaluation Metrics

Used:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**
- `classification_report` from `sklearn`

### Confusion Matrix After SMOTE:

|               | Predicted 0 | Predicted 1 |
|---------------|-------------|-------------|
| **Actual 0**  | 958         | 2           |
| **Actual 1**  | 16          | 139         |

### Final Performance (on Test Set)

| Metric    | Class 0 | Class 1 |
|-----------|---------|---------|
| Precision | 0.98    | 0.99    |
| Recall    | 1.00    | 0.90    |
| F1-score  | 0.99    | 0.94    |

---

##  Libraries Used

- `numpy`
- `pandas`
- `scikit-learn`
  - `LogisticRegression`
  - `SMOTE` (from `imblearn.over_sampling`)
  - `classification_report`, `confusion_matrix`, `accuracy_score`

---

## 📈 Results

- **Initial accuracy**: ~97% (biased toward majority class).
- **After class weighting**: Improved minority class recall.
- **After SMOTE**: Further improved performance, especially recall and F1-score of the minority class.
- **Final accuracy**: ~98% with balanced precision and recall.

---

## Future Improvements

- Try more complex classifiers (e.g., Random Forest, XGBoost).
- Use GridSearchCV for hyperparameter tuning.
- Evaluate performance using ROC-AUC.

---

##  Author

- **Harsh Kumar** – Computer Science Student  
  Interested in machine learning, competitive programming, and coffee ☕ + code 💻

---

## Note

**Never apply SMOTE to the test data!**  
SMOTE is only applied to the **training data** to avoid data leakage.

---  

## 💬 Feel free to fork, raise issues, or contribute :)


# Heart Disease Prediction Using Machine Learning

## 📋 Project Overview
This project implements a machine learning solution to predict heart disease in patients using clinical and demographic features. The model performs binary classification to assist healthcare professionals in early detection.

## 🎯 Objective
Develop an accurate machine learning model that can predict heart disease presence based on patient characteristics and clinical measurements.

## 📊 Dataset Information
- **Size**: 7,303 patient records with 15 variables
- **Data Quality**: Complete dataset with zero missing values and no duplicates
- **Target Distribution**: 80% positive cases (heart disease present)

### Key Features
- **Demographics**: Age, gender
- **Clinical Measurements**: Blood pressure, cholesterol, max heart rate
- **Symptoms**: Chest pain type, exercise-induced angina
- **Diagnostic Tests**: ECG results, ST depression, vessel count, thalassemia

## 🔧 Data Preprocessing

### Feature Engineering
- **Ordinal Encoding**: Applied to chest pain, ST slope, vessel count
- **One-Hot Encoding**: Applied to nominal categories (gender, blood sugar, etc.)
- **Scaling**: StandardScaler applied to continuous variables
- **Balancing**: SMOTE used to address class imbalance

## 🤖 Machine Learning Models

### Models Implemented
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- CatBoost
- XGBoost
- LightGBM

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization

## 📈 Key Findings
- **Strong predictors**: Exercise-induced angina and vessel count show highest discrimination
- **Balanced features**: No extreme category imbalances detected
- **Quality data**: All values within medically realistic ranges
- **Optimal age range**: 29-77 years (avg: 53.2) ideal for prediction models

## 🛠️ Technologies Used
```python
# Core Libraries
pandas, numpy, matplotlib, seaborn
scikit-learn, imbalanced-learn
xgboost, catboost, lightgbm, SMOTE
```

## 🚀 Getting Started

### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### Usage
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
jupyter notebook heart_disease_prediction.ipynb
```

## 📊 Results

### Model Performance Comparison
| Model | Train Accuracy | Test Accuracy | Test Recall | Test Precision | AUC | Clinical Assessment |
|-------|----------------|---------------|-------------|----------------|-----|-------------------|
| **Logistic Regression** ✅ | 85.4% | 81.9% | 84.4% | 92.9% | 89% | **SELECTED** |
| Decision Tree | 100% | 82.7% | 87.3% | 91.2% | 74% | Overfitted |
| XGBoost | 99.3% | 82.1% | 86.3% | 91.4% | 89% | Overfitted |

## 🔍 Future Enhancements
- Ensemble methods implementation
- Hyperparameter optimization
- Model deployment
- Cross-validation analysis

## 📞 Contact
**adescofaj**  
Email: adescofaj@gmail.com  
GitHub: https://github.com/adescofaj

---
*Machine learning for better healthcare outcomes*
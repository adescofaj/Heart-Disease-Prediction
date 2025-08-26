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
- Random Forest (planned)
- XGBoost (planned)

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
| Model | Train Accuracy | Test Accuracy | F1-Score |
|-------|----------------|---------------|----------|
| Logistic Regression | TBD | TBD | TBD |
| SVM | TBD | TBD | TBD |

## 🔍 Future Enhancements
- Ensemble methods implementation
- Hyperparameter optimization
- Model deployment
- Cross-validation analysis

## 📄 License
MIT License

## 📞 Contact
**Your Name**  
Email: your.email@example.com  
GitHub: [@yourusername](https://github.com/yourusername)

---
*Machine learning for better healthcare outcomes*
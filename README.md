# HDB-Resale-Price-Prediction
Machine learning project predicting HDB resale prices using regression models and model evaluation techniques.

# HDB Resale Price Prediction

## Overview

This project applies machine learning regression techniques to predict HDB resale flat prices in Singapore. The objective is to identify key factors influencing resale prices and compare different regression models to determine the most suitable approach for housing price prediction.

The project includes data preprocessing, exploratory data analysis (EDA), feature selection, model training, performance evaluation, and model comparison.

---

## Objectives

- Analyse key factors affecting HDB resale prices  
- Implement Multiple Linear Regression as a baseline model  
- Implement KNN Regression for non-linear modelling  
- Compare model performance using regression evaluation metrics  
- Identify the best-performing model  

---

## Dataset

The dataset contains HDB resale flat transaction records, including features such as:

- Floor area (sqm)  
- Remaining lease  
- Flat age  
- Flat type  
- Town  
- Transaction year  
- Resale price (target variable)  

Data preprocessing steps include:
- Handling missing values  
- Encoding categorical variables  
- Feature scaling (for KNN)  

---

## Models Implemented

### 1. Multiple Linear Regression (MLR)
- Used as the baseline model
- Assumes a linear relationship between features and resale price
- Provides interpretability of feature influence

### 2. K-Nearest Neighbours (KNN) Regression
- Captures non-linear relationships
- Uses similarity-based prediction
- Performance depends on selected value of K and feature scaling

---

## Model Evaluation

Models were evaluated using regression metrics such as:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
- Adjusted R-squared

### Final Result

KNN Regression achieved better predictive performance compared to Multiple Linear Regression, indicating that resale prices exhibit non-linear patterns that are better captured using distance-based methods.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

---

## Project Structure
├── AIML_Project_HDBResale1.ipynb
├── dataset/
├── README.md

---

## Key Learning Outcomes

- Understanding linear vs non-linear regression modelling
- Feature engineering and data preprocessing
- Model comparison and evaluation
- Selecting appropriate models based on data characteristics

---

## Future Improvements

- Implement advanced regression models (Random Forest, XGBoost)
- Perform hyperparameter tuning for KNN
- Deploy model using a web interface
- Use cross-validation for more robust evaluation

---

## Author

Kenneth  
AI & Machine Learning Project

# churn_predictor.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

def load_data(path='customer_churn.csv'):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    # Convert TotalCharges from string to numeric if present
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    
    # Encode categorical features
    le = LabelEncoder()
    for col in df.select_dtypes(include='object'):
        if df[col].nunique() == 2:
            df[col] = le.fit_transform(df[col])
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    
    df.drop(columns=['customerID'], errors='ignore', inplace=True)
    return df

def split_resample(df, target_col='Churn'):
    X = df.drop(columns=[target_col])
    y = df[target_col].apply(lambda x: 1 if x in [1, 'Yes'] else 0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    return X_train_bal, X_test, y_train_bal, y_test

def train_models(X_train, y_train):
    models = dict()
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    models['LogisticRegression'] = lr.fit(X_train, y_train)
    # Random Forest with simple tuning
    rf = RandomForestClassifier(random_state=42)
    params = {'n_estimators': [100, 200], 'max_depth': [None, 10]}
    grid = GridSearchCV(rf, params, cv=3, scoring='f1', n_jobs=-1)
    models['RandomForest'] = grid.fit(X_train, y_train).best_estimator_
    return models

def evaluate(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        print(f"\n*** Model: {name} ***")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print("F1 Score:", f1_score(y_test, y_pred))
        print("ROC AUC:", roc_auc_score(y_test, y_proba))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix — {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

def plot_feature_importance(model, feature_names, top_n=10):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
        plt.title('Top Feature Importances')
        plt.tight_layout()
        plt.show()

def main():
    df = load_data('customer_churn.csv')
    df = preprocess(df)
    X_train, X_test, y_train, y_test = split_resample(df)
    models = train_models(X_train, y_train)
    evaluate(models, X_test, y_test)
    # Plot feature importances for RandomForest
    plot_feature_importance(models['RandomForest'], X_train.columns)

if __name__ == '__main__':
    main()

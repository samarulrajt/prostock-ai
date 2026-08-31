import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_model():
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv('customer_data.csv')
    
    # 2. Preprocessing
    print("Preprocessing data...")
    # Drop ID as it's not a feature
    df = df.drop('customer_id', axis=1)
    
    # --- Feature Engineering ---
    # Create a feature: Average charge per month of tenure
    # Avoid division by zero by adding a small epsilon
    df['charge_per_tenure'] = df['total_charges'] / (df['tenure'] + 1)
    
    # Encode categorical data
    le = LabelEncoder()
    df['contract_type'] = le.fit_transform(df['contract_type'])
    
    # Split features and target
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Split into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 3. Model Selection & Training
    print("Training Random Forest model...")
    # Optimized hyperparameters: 
    # - n_estimators=200 for better stability
    # - max_depth=10 to prevent overfitting
    # - min_samples_split=5 to ensure nodes are meaningful
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        min_samples_split=5, 
        random_state=42, 
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # 4. Evaluation
    print("\n--- Model Evaluation ---")
    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))

    # --- Feature Importance ---
    print("\n--- Feature Importance ---")
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_df.to_string(index=False))

if __name__ == "__main__":
    train_model()

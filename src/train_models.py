import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

def train_and_export_model():
    print("Initializing model training pipeline...")

    # 1. Load the data
    data_path = 'data/Diabetes_and_LifeStyle_Dataset_.csv'
    if not os.path.exists(data_path):
        print(f"Error: Could not find dataset at {data_path}. Ensure you are running the script from the project root.")
        return

    df = pd.read_csv(data_path)

    # 2. Prevent Data Leakage and separate Target/Features
    df_clean = df.drop(columns=['diabetes_risk_score', 'diagnosed_diabetes'])
    X_raw = df_clean.drop(columns=['diabetes_stage'])
    y_raw = df_clean['diabetes_stage']

    # 3. Encode Target and Features
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    # One-hot encoding features
    X = pd.get_dummies(X_raw, drop_first=True)
    
    # Store the feature names - critical for Phase 6 Web App
    model_features = X.columns.tolist()

    # 4. Train-Test Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Compute weights for handling class imbalance
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # 6. Initialize and Train the Final XGBoost Model
    print("Training XGBoost Classifier...")
    xgb_model = XGBClassifier(
        random_state=42, 
        eval_metric='mlogloss',
        max_depth=6,          
        learning_rate=0.1,    
        n_estimators=200      
    )
    
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

    # 7. Evaluate the Model
    y_pred = xgb_model.predict(X_test)
    print("\n--- Final Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Added zero_division=0 to keep console output clean
    # Added .astype(str) to prevent LabelEncoder type errors
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_.astype(str), 
        zero_division=0
    ))

    # 8. Export Artifacts
    os.makedirs('artifacts', exist_ok=True)
    
    # Save the model
    joblib.dump(xgb_model, 'artifacts/model_1.pkl')
    
    # Save the label encoder (to turn numbers back into "Type 2", etc.)
    joblib.dump(label_encoder, 'artifacts/label_encoder.pkl')
    
    # Save the feature list (so the Web App knows the column order)
    joblib.dump(model_features, 'artifacts/model_features.pkl')
    
    print("\nSuccess! All artifacts saved to /artifacts directory.")

if __name__ == '__main__':
    train_and_export_model()
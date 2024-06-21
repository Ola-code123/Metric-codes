import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
import shap

# Paths to the dataset files
train_path = '...'
test_path = '...'


train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


cat_cols = ['proto', 'service', 'state', 'attack_cat']

# Encoding
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

train_data[cat_cols] = encoder.fit_transform(train_data[cat_cols])
test_data[cat_cols] = encoder.transform(test_data[cat_cols])

# Drop columns
X_training = train_data.drop(['id', 'label', 'attack_cat'], axis=1)
y_training = train_data['label']
X_testing = test_data.drop(['id', 'label', 'attack_cat'], axis=1)
y_testing = test_data['label']

# Random Forest model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_training, y_training)

# Sample 300 instances for SHAP analysis
np.random.seed(42)
sample_indices = np.random.choice(X_testing.index, 300, replace=False)
one_index = sample_indices[0]  
X_one_instance = X_testing.loc[one_index:one_index]

# Starting SHAP
explainer = shap.TreeExplainer(model)
shap_values_instance = explainer.shap_values(X_one_instance)

# Get the top features from SHAP values
feature_importance = np.abs(shap_values_instance[1]).flatten()
top_features_indices = np.argsort(-feature_importance)
top_3_features = X_testing.columns[top_features_indices[:3]].tolist()
top_5_features = X_testing.columns[top_features_indices[:5]].tolist()

# Function to evaluate model accuracy by removing features
def model_accuracy_evaluation(features_to_remove):
    X_training_reduced = X_training.drop(columns=features_to_remove)
    X_testing_reduced = X_testing.drop(columns=features_to_remove)

    # Retrain the model with the reduced feature set
    reduced_model = RandomForestClassifier(n_estimators=100, random_state=42)
    reduced_model.fit(X_training_reduced, y_training)
    reduced_accuracy = reduced_model.score(X_testing_reduced, y_testing)
    
    print(f'Features Removed: {features_to_remove}')
    print(f'Accuracy with Features Removed: {reduced_accuracy:.4f}')

# Evaluating model accuracy without removing features, with top 3 and top 5 features removed
model_accuracy_evaluation([])
model_accuracy_evaluation(top_3_features)
model_accuracy_evaluation(top_5_features)

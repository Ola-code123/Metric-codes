import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
import shap

# Add Paths to the NSL-KDD train and test datasets
train_path = '...'
test_path = '...'

# Column names for the datasets
col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
             "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

# Load training and testing data
train_data = pd.read_csv(train_path, header=None, names=col_names)
test_data = pd.read_csv(test_path, header=None, names=col_names)

cat_cols = ["protocol_type", "service", "flag"]

# Encoding with OrdinalEncoder
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_data[cat_cols] = encoder.fit_transform(train_data[cat_cols])
test_data[cat_cols] = encoder.transform(test_data[cat_cols])

# Prepare datasets with target labels
X_training = train_data.drop('label', axis=1)
y_training = train_data['label']
X_testing = test_data.drop('label', axis=1)
y_testing = test_data['label']

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_training, y_training)

# Sample 300 instances for SHAP analysis
np.random.seed(42)
sample_indices = np.random.choice(X_testing.index, 300, replace=False)
one_index = sample_indices[0]  
X_one_instance = X_testing.loc[one_index:one_index]

# Initialize SHAP
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

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from lime.lime_tabular import LimeTabularExplainer
import re

# Paths to the dataset files
train_path = '...'
test_path = '...'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

cat_cols = ['proto', 'service', 'state', 'attack_cat']

# Encoding
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder.fit(train_data[cat_cols])  
train_data[cat_cols] = encoder.transform(train_data[cat_cols])  
test_data[cat_cols] = encoder.transform(test_data[cat_cols])  

# inverse mappings 
inverse_mappings = {}
for idx, col in enumerate(cat_cols):
    # Map encoded values back to original 
    categories = encoder.categories_[idx]
    inverse_mappings[col] = {i: categories[i] for i in range(len(categories))}

    
# Function to map encoded values back to original values
def decode_feature_name(feature_name):
    for col, mapping in inverse_mappings.items():
        if feature_name.isdigit() and int(feature_name) in mapping:
            return f"{col}_{mapping[int(feature_name)]}"
    return feature_name

# Drop columns
X_training = train_data.drop(['id', 'label', 'attack_cat'], axis=1)
y_training = train_data['label']
X_testing = test_data.drop(['id', 'label', 'attack_cat'], axis=1)
y_testing = test_data['label']

# Random Forest model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_training, y_training)

# Sample of 300 instances from the testing data for LIME explanations
np.random.seed(42)
sample_indices = np.random.choice(X_testing.shape[0], 300, replace=False)
X_testing_sample = X_testing.iloc[sample_indices]
y_testing_sample = y_testing.iloc[sample_indices]

# Starting LIME explainer with the sampled testing data
explainer = LimeTabularExplainer(X_testing_sample.values,
                                 feature_names=X_testing_sample.columns.tolist(),
                                 class_names=['Non-Attack', 'Attack'],
                                 mode='classification')

# Function to evaluate model accuracy by removing features
def model_accuracy_evaluation(num_features):
    # Explain an instance from the test set with LIME
    exp = explainer.explain_instance(X_testing.iloc[0].values, model.predict_proba, num_features=num_features)
    important_features = []
    
    # Extract feature names correctly from the LIME explanation
    for feature_description, weight in exp.as_list():
        match = re.match(r"^[^\d\s]+", feature_description)
        if match:
            feature_name = match.group(0)
            decoded_name = decode_feature_name(feature_name)
            important_features.append(decoded_name)
    
    # Remove the important features from train and test datasets
    X_training_removed = X_training.drop(columns=important_features, errors='ignore')
    X_testing_removed = X_testing.drop(columns=important_features, errors='ignore')

    # Retrain the model with removed features
    model_retrained = RandomForestClassifier(random_state=42)
    model_retrained.fit(X_training_removed, y_training)
    removed_accuracy = model_retrained.score(X_testing_removed, y_testing)

    print(f'Removed Top {num_features} Features: {important_features[:num_features]}')
    print(f'Accuracy with Top {num_features} Features Removed: {removed_accuracy}')


# Evaluate model accuracy with no features removed and with the top 3 and top 5 features removed
original_accuracy = model.score(X_testing, y_testing)
print(f'Original Model Accuracy: {original_accuracy}')
model_accuracy_evaluation(3) 
model_accuracy_evaluation(5)

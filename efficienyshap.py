import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import time

# Add the paths to the UNSW-NB15 train and test dataset
train_path = '...'
test_path = '...'

# Read the training and testing data
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Combine datasets for preprocessing
combined_data = pd.concat([train_data, test_data])

# Label encoding for categorical columns 
cat_cols = ['proto', 'service', 'state', 'attack_cat']
for col in cat_cols:
    le = LabelEncoder()
    combined_data[col] = le.fit_transform(combined_data[col])

# Split the data back into train and test
train_data = combined_data.iloc[:len(train_data)]
test_data = combined_data.iloc[len(train_data):]

# Drop columns not used in the analysis and prepare labels
X_training = train_data.drop(['id', 'label', 'attack_cat'], axis=1)
y_training = train_data['label']
X_testing = test_data.drop(['id', 'label', 'attack_cat'], axis=1)
y_testing = test_data['label']

# Initialize and train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_training, y_training)

# Initialize SHAP explainer on the trained Random Forest model
explainer = shap.TreeExplainer(model)

# Sample sizes
sample_sizes = [10, 100, 1000]

# Iterate over sample sizes to compute SHAP values and measure performance
for size in sample_sizes:
    if size <= len(X_testing):
        # Select a random sample from the test data
        sample_test = X_testing.sample(n=size, random_state=42)
        start_time = time.time()
         # Compute SHAP values for the sample
        _ = explainer.shap_values(sample_test)  

        # Calculate and print the time taken to compute SHAP values
        shap_time = time.time() - start_time
        print(f"SHAP explanations for {size} samples computed in {shap_time:.2f} seconds")
    else:
         # Handle case where the sample size exceeds available data
        print(f"Sample size {size} exceeds the number of available samples {len(X_testing)}.")

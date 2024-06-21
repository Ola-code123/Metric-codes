import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import time

# Paths to the dataset files
train_path = '...'
test_path = '...'


train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Combine datasets for preprocessing
combined_data = pd.concat([train_data, test_data])

# Label encoding 
cat_cols = ['proto', 'service', 'state', 'attack_cat']
for col in cat_cols:
    le = LabelEncoder()
    combined_data[col] = le.fit_transform(combined_data[col])

# Split the data back into train and test
train_data = combined_data.iloc[:len(train_data)]
test_data = combined_data.iloc[len(train_data):]

# Drop columns
X_training = train_data.drop(['id', 'label', 'attack_cat'], axis=1)
y_training = train_data['label']
X_testing = test_data.drop(['id', 'label', 'attack_cat'], axis=1)
y_testing = test_data['label']

# Random Forest model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_training, y_training)

# Starting SHAP
explainer = shap.TreeExplainer(model)

# Sample sizes
sample_sizes = [10, 100, 1000]

# Iterate over sample sizes and compute SHAP explaining time
for size in sample_sizes:
    if size <= len(X_testing):
        sample_test = X_testing.sample(n=size, random_state=42)
        start_time = time.time()
        _ = explainer.shap_values(sample_test)  
        
        shap_time = time.time() - start_time
        print(f"SHAP explanations for {size} samples computed in {shap_time:.2f} seconds")
    else:
        print(f"Sample size {size} exceeds the number of available samples {len(X_testing)}.")

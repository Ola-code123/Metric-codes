import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from lime import lime_tabular
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

# Sample sizes 
sample_sizes = [10, 100, 1000]

# Starting LIME explainer
explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_training.values,
    feature_names=X_training.columns.tolist(),
    class_names=['Normal', 'Attack'],
    mode='classification'
)

# Iterate over each sample size for LIME explanations
for size in sample_sizes:
    if size <= len(X_testing):
        sample_test = X_testing.sample(n=size, random_state=42)
        start_time = time.time()
        for i in range(size):
            exp = explainer.explain_instance(sample_test.iloc[i].values, model.predict_proba, num_features=3)
        lime_time = time.time() - start_time
        print(f"LIME explanation for all {size} samples computed in {lime_time:.2f} seconds")
    else:
        print(f"Sample size {size} exceeds the number of available samples {len(X_testing)}.")

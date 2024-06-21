import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import time

# Paths to the dataset files
train_path = '...'
test_path = '...'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Defining the categorical columns
cat_cols = ['proto', 'service', 'state']

# Encoding
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_data[cat_cols] = encoder.fit_transform(train_data[cat_cols])
test_data[cat_cols] = encoder.transform(test_data[cat_cols])

# Defining the features to be used
features_reduced = ['spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 
                   'sloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 
                   'smean', 'dmean', 'response_body_len']


X_training = train_data[features_reduced]
y_training = train_data['label']
X_testing = test_data[features_reduced]
y_testing = test_data['label']

# Training the RandomForest model using the reduced features
model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1)
model.fit(X_training, y_training)

# Start LIME explainer 
explainer = LimeTabularExplainer(
    training_data=X_training.values,
    feature_names=features_reduced,
    class_names=['Not Attack', 'Attack'],
    categorical_features=[X_training.columns.get_loc(c) for c in cat_cols if c in features_reduced],
    mode='classification'
)

# Stability analysis 
N = 3  # Number of runs
k = 10  # Top-k features
sample_size = 50  # Size of sample for each explanation run

# Collecting top-k features across N runs
top_features_sets = []
for i in range(N):
    np.random.seed(i)
    sample_indices = np.random.choice(X_testing.index, size=sample_size, replace=False)
    sample = X_testing.iloc[sample_indices]
    lime_features = []
    for idx in range(sample_size):
        exp = explainer.explain_instance(sample.iloc[idx].values, model.predict_proba, num_features=k)
        features = set(re.match(r"^(.*?)(\s*<=|<|=|>|>=)\s*[\d.+-]+", feat[0]).group(1) for feat in exp.as_list())
        lime_features.extend(features)
    top_features_sets.append(set(lime_features[:k]))

# Calculate stability
common_features = set.intersection(*top_features_sets)
stability_score = len(common_features) / k
print(f"Stability Score: {stability_score}")
print("Number of Common Features:", len(common_features))
print("Common Features:", ', '.join(common_features))

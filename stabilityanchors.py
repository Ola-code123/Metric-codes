import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from alibi.explainers import AnchorTabular
import matplotlib.pyplot as plt

# Add paths to the UNSW-NB15 train and test dataset files
train_path = '...'
test_path = '...'

# Load datasets
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


cat_cols = ['proto', 'service', 'state']

# Defining features for the model
features_reduced = ['spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 
                   'sloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 
                   'smean', 'dmean', 'response_body_len']

# Encoding with OrdinalEncoder
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_data[cat_cols] = encoder.fit_transform(train_data[cat_cols])
test_data[cat_cols] = encoder.transform(test_data[cat_cols])

# Prepare datasets
X_training = train_data[features_reduced]
y_training = train_data['label']
X_testing = test_data[features_reduced]
y_testing = test_data['label']

# Training the Random Forest model using the reduced features
model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1)
model.fit(X_training, y_training)

# Initialize the AnchorTabular explainer 
explainer = AnchorTabular(model.predict_proba, feature_names=features_reduced)
explainer.fit(X_training.values, disc_perc=[10, 50, 90])  

# Stability analysis 
N = 3
k = 10
sample_size = 50
threshold = 0.70

# Collecting top-k anchors across N runs
top_anchors_sets = []
for i in range(N):
    indices = np.random.choice(X_testing.index, size=sample_size, replace=False)
    anchor_features = []
    for idx in indices:
        sample_instance = X_testing.iloc[idx].values.reshape(1, -1)  
        explanation = explainer.explain(sample_instance, threshold=threshold)
        if explanation.anchor:
            anchor_features.append(' AND '.join(explanation.anchor))
    top_anchors_sets.append(set(anchor_features[:k]))

# Calculate stability
common_anchors = set.intersection(*top_anchors_sets)
stability_score = len(common_anchors) / k
print(f"Stability Score: {stability_score}")
print("Number of Common Anchors:", len(common_anchors))
print("Common Anchors:", ', '.join(common_anchors))

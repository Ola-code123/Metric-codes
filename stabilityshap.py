import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
import shap
import matplotlib.pyplot as plt

# Paths to the dataset files
train_path = '...'
test_path = '...'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Encoding
cat_cols = ['proto', 'service', 'state']
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

# Starting SHAP explainer 
explainer = shap.TreeExplainer(model)

# Collecting top-k features across N runs
top_features_sets = []
for i in range(3):  
    np.random.seed(i)
    sample_indices = np.random.choice(X_testing.index, size=50, replace=False)  
    for idx in sample_indices:
        shap_values = explainer.shap_values(X_testing.iloc[idx:idx+1])
        feature_importance = np.abs(shap_values[1]).flatten()
        top_indices = np.argsort(feature_importance)[-10:]  
        top_features = set(X_testing.columns[top_indices])
        top_features_sets.append(top_features)  
        print(f"Run {i}, Instance {idx}, Top Features: {top_features}")


# Calculate stability 
if top_features_sets:  
    common_features = set.intersection(*top_features_sets)
    stability_score = len(common_features) / 10
    print(f"Stability Score: {stability_score}")
    print("Common features across runs:", ', '.join(common_features))
else:
    print("No common features")

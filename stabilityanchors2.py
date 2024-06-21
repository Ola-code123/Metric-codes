import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from alibi.explainers import AnchorTabular
import time

# Column names for the NSL-KDD datasets
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

# Add paths to the NSL-KDD train and test datasets 
train_path= '...'
test_path = '...'

# Load datasets
train_data = pd.read_csv(train_path, names=col_names)
test_data = pd.read_csv(test_path, names=col_names)

# Encoding with LabelEncoder
label_encoder = LabelEncoder()
categorical_cols = ['protocol_type', 'service', 'flag']  
for col in categorical_cols:
    label_encoder.fit(pd.concat([train_data[col], test_data[col]]))
    train_data[col] = label_encoder.transform(train_data[col])
    test_data[col] = label_encoder.transform(test_data[col])

# Defining features for the model
features_reduced = ['duration', 'service', 'flag', 'src_bytes', 'dst_bytes', 'hot', 'logged_in', 
                   'num_compromised', 'num_root', 'num_file_creations', 'count', 'srv_count', 
                   'serror_rate', 'srv_serror_rate', 'same_srv_rate', 'dst_host_count', 
                   'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_serror_rate', 
                   'dst_host_srv_serror_rate']

# Prepare datasets
X_training = train_data[features_reduced]
y_training = train_data['label']
X_testing = test_data[features_reduced]
y_testing = test_data['label']

# Training the RandomForest model using the reduced features
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

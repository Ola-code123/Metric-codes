import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from alibi.explainers import AnchorTabular
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


# Initialize Anchors 
explainer = AnchorTabular(predictor=model.predict_proba, feature_names=X_training.columns.tolist())
explainer.fit(X_training.values, disc_perc=(25, 50, 75))

# sample sizes
sample_sizes = [10, 100, 1000]

# Iterate over each sample size for explanations
for size in sample_sizes:
    if size <= len(X_testing):
        sample_test = X_testing.sample(n=size, random_state=42)
        start_time = time.time()
        explanations = []
        for i in range(size):
            # Generate an explanation for each instance
            explanation = explainer.explain(sample_test.iloc[i].values, threshold=0.95)
            explanations.append(explanation.anchor)
            # Calculate and print the time taken to compute Anchors
        anchor_time = time.time() - start_time
        print(f"Anchor explanation for {size} samples computed in {anchor_time:.2f} seconds")
    else:
        # Handle case where the sample size exceeds available data
        print(f"Sample size {size} exceeds the number of available samples {len(X_testing)}.")

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from alibi.explainers import AnchorTabular

# Paths to the dataset files
train_path = '...'
test_path = '...'


train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


cat_cols = ['proto', 'service', 'state', 'attack_cat']

# Encoding
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

train_data[cat_cols] = encoder.fit_transform(train_data[cat_cols])
test_data[cat_cols] = encoder.transform(test_data[cat_cols])

# Drop columns
X_training = train_data.drop(['id', 'label', 'attack_cat'], axis=1)
y_training = train_data['label']
X_testing = test_data.drop(['id', 'label', 'attack_cat'], axis=1)
y_testing = test_data['label']

# Random Forest model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_training, y_training)

# Sample 300 instances from the testing data for Anchor explanations
np.random.seed(42)
sample_indices = np.random.choice(X_testing.shape[0], 300, replace=False)
X_testing_sample = X_testing.iloc[sample_indices]

# Function to use with AnchorTabular for predictions
def predict_function(X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=X_testing.columns)
    return model.predict(X)

# Starting the Anchor explainer
explainer = AnchorTabular(predict_function, feature_names=X_testing.columns.tolist())
explainer.fit(X_testing_sample.values, disc_perc=[25, 50, 75])

# Function to evaluate model accuracy by removing features
def model_accuracy_evaluation(num_features):
    # Explain an instance using Anchor and get the important features
    explanation = explainer.explain(X_testing.values[0], threshold=0.95)
    # Extracting feature names 
    anchor_features = [re.split(' < | > | <= | >= | = ', name)[0].strip() for name in explanation.anchor][:num_features]

   # Remove the important features from train and test datasets
    X_training_removed = X_training.drop(columns=anchor_features, errors='ignore')
    X_testing_removed = X_testing.drop(columns=anchor_features, errors='ignore')

    # Retrain the model with removed features
    model_retrained = RandomForestClassifier(random_state=42)
    model_retrained.fit(X_training_removed, y_training)
    removed_accuracy = model_retrained.score(X_testing_removed, y_testing)

    
    print(f'Removed Top {num_features} Anchor Features: {anchor_features}')
    print(f'Accuracy with Top {num_features} Anchor Features Removed: {removed_accuracy}')

# Evaluate model accuracy with no features removed and with the top 3 and top 5 features removed
original_accuracy = model.score(X_testing, y_testing)
print(f'Original Model Accuracy: {original_accuracy}')

model_accuracy_evaluation(3)
model_accuracy_evaluation(5)

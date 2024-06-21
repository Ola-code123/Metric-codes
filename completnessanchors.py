import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from alibi.explainers import AnchorTabular

# Add paths to UNSW-NB15 train and test datasets
train_path = '...'
test_path = '...'

# Load datasets
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Mapping for attack categories 
attack_categories = {
    0: 'Normal',
    1: 'Fuzzers',
    2: 'Analysis',
    3: 'Backdoor',
    4: 'DoS',
    5: 'Exploits',
    6: 'Generic',
    7: 'Reconnaissance',
    8: 'Shellcode',
    9: 'Worms'
}

# Drop 'id' column 
if 'id' in train_data.columns:
    train_data.drop('id', axis=1, inplace=True)
    test_data.drop('id', axis=1, inplace=True)

# Combine test and train dataset for preprocessing
combined_data = pd.concat([train_data, test_data])

# Encoding with LabeEncoder
categorical_cols = combined_data.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined_data[col] = le.fit_transform(combined_data[col])
    label_encoders[col] = le

# Split back into training and testing sets
train_data = combined_data.iloc[:len(train_data)]
test_data = combined_data.iloc[len(train_data):]


feature_names = train_data.drop(['label', 'attack_cat'], axis=1).columns.tolist()

# Prepare datasets
X_training = train_data.drop(['label', 'attack_cat'], axis=1).values
y_train_attack_cat = train_data['attack_cat'].astype(int)
X_testing = test_data.drop(['label', 'attack_cat'], axis=1).values
y_test_attack_cat = test_data['attack_cat'].astype(int)

# Scale features
scaler = StandardScaler()
X_training = scaler.fit_transform(X_training)
X_testing = scaler.transform(X_testing)

# Training model for predicting attack categories
attack_cat_model = RandomForestClassifier(n_estimators=100, random_state=42)
attack_cat_model.fit(X_training, y_train_attack_cat)

# Intialize Anchor explainer
explainer = AnchorTabular(predictor=attack_cat_model.predict, feature_names=feature_names)
explainer.fit(X_training)

# Plotting the feature importances
def plot_feature_importances(features, values, perturbation):
    plt.figure(figsize=(10, 5))
    plt.bar(features, values, color='skyblue')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.title(f'Feature Values at Perturbation {perturbation:.1f}')
    plt.xticks(rotation=45, ha='right')  
    plt.tight_layout()
    plt.show()

# Plotting the perturbation effects
def plot_perturbation_effects(perturbation_results):
    perturbations = [result[0] for result in perturbation_results]
    precisions = [result[2] for result in perturbation_results]
    coverages = [result[3] for result in perturbation_results]
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(perturbations, precisions, 'b-', label='Precision')
    ax1.set_xlabel('Perturbation Level')
    ax1.set_ylabel('Precision', color='b')
    ax1.tick_params('y', colors='b')
    
    ax2 = ax1.twinx()
    ax2.plot(perturbations, coverages, 'r--', label='Coverage')
    ax2.set_ylabel('Coverage', color='r')
    ax2.tick_params('y', colors='r')
    
    plt.title('Perturbation Effects on Model Predictions')
    fig.tight_layout()
    plt.legend(loc='upper right')
    plt.show()

# Function to analyze how feature importance changes with perturbation 
def analyze_completeness(sample_index, threshold=0.75):
    # Generate a single sample and its original attack category
    sample = X_testing[sample_index]
    original_prediction = attack_cat_model.predict([sample])[0]
    original_attack_cat = attack_categories.get(original_prediction, 'Unknown')
    print(f"Original attack category: {original_attack_cat} (Code: {original_prediction})\n")

    # Generate an explanation for the sample
    explanation = explainer.explain(sample, threshold=threshold)
    print("Initial Anchor:", explanation.anchor if explanation.anchor else "No anchor formed")
    print("Initial Precision: %.2f" % explanation.precision)
    print("Initial Coverage: %.2f" % explanation.coverage)
    
    perturbation_results = []
    
    if explanation.anchor:
        # Extract feature names from the anchor conditions
        feature_indices = []
        for condition in explanation.anchor:
            feature_name = condition.split(' ')[0]  
            if feature_name in feature_names:
                feature_indices.append(feature_names.index(feature_name))
            else:
                print(f"Feature name '{feature_name}' from anchor '{condition}' not found in feature list.")
                continue

        # Perturb features and analyze impact
        for perturbation in np.linspace(0, 1, 11):
            perturbed_sample = np.array(sample, copy=True)
            current_values = []
            
            for feature_idx in feature_indices:
                original_value = perturbed_sample[feature_idx]
                perturbed_sample[feature_idx] = original_value * (1 - perturbation)
                current_values.append(perturbed_sample[feature_idx])
            
            perturbed_prediction = attack_cat_model.predict([perturbed_sample])[0]
            new_attack_cat = attack_categories.get(perturbed_prediction, 'Unknown')
            new_explanation = explainer.explain(perturbed_sample, threshold=threshold)
            
            print(f"Perturbation {perturbation:.1f}: Attack category changes to {new_attack_cat} (Code: {perturbed_prediction})")
            print("New Anchor:", new_explanation.anchor if new_explanation.anchor else "No anchor formed")
            print("Top Features and their values after perturbation:")
            for idx, value in zip(feature_indices, current_values):
                print(f"    {feature_names[idx]}: {value:.2f}")
            
            plot_feature_importances([feature_names[idx] for idx in feature_indices], current_values, perturbation)

            perturbation_results.append((perturbation, new_attack_cat, new_explanation.precision, new_explanation.coverage))

            if perturbed_prediction != original_prediction:
                print(f"Stopping as the prediction changed from {original_attack_cat} to {new_attack_cat} at perturbation {perturbation:.1f}")
                break

        plot_perturbation_effects(perturbation_results)

# Analyzing first sample
analyze_completeness(0, threshold=0.75)


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap
import matplotlib.pyplot as plt

# Paths to the datasets
train_path = '...'
test_path = '...'
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
combined_data = pd.concat([train_data, test_data])

# Drop 'id' from dataset
combined_data.drop('id', axis=1, inplace=True)

# Encoding variables
category_cols = combined_data.select_dtypes(include=['object']).columns
label_encoders = {}
for col in category_cols:
    le = LabelEncoder()
    combined_data[col] = le.fit_transform(combined_data[col])
    label_encoders[col] = le

# Split back into training and testing sets
train_data = combined_data.iloc[:len(train_data)]
test_data = combined_data.iloc[len(train_data):]

# Drop columns
feature_names = train_data.drop(['label', 'attack_cat'], axis=1).columns.tolist()
X_training = train_data.drop(['label', 'attack_cat'], axis=1).values
y_training_attack_cat = train_data['attack_cat'].values
X_testing = test_data.drop(['label', 'attack_cat'], axis=1).values
y_testing_attack_cat = test_data['attack_cat'].values

# Feature scaling
scaler = StandardScaler()
X_training = scaler.fit_transform(X_training)
X_testing = scaler.transform(X_testing)

# Train model for predicting attack categories
attack_cat_model = RandomForestClassifier(n_estimators=100, random_state=42)
attack_cat_model.fit(X_training, y_training_attack_cat)

# SHAP Explainer
explainer = shap.TreeExplainer(attack_cat_model)
random_indices = np.random.choice(X_training.shape[0], 300, replace=False)
sample_X_testing = X_training[random_indices]
shap_values_subset = explainer.shap_values(sample_X_testing)
shap.summary_plot(shap_values_subset, sample_X_testing, plot_type="bar")

# Attack category mapping
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

def analyze_completeness(sample_index, top_n_features=5):
    sample = X_testing[sample_index]
    original_attack_cat_code = attack_cat_model.predict([sample])[0]
    original_attack_cat = attack_categories[original_attack_cat_code]
    original_shap_values = explainer.shap_values(sample)
    feature_importance = np.argsort(np.abs(original_shap_values).mean(0))[::-1][:top_n_features]
    
    print(f"Original attack category: {original_attack_cat} (Code: {original_attack_cat_code})\n")
    print("Top initial feature importances:")
    initial_feature_importances = {feature_names[idx]: np.abs(original_shap_values).mean(0)[idx] for idx in feature_importance}
    for feature, importance in sorted(initial_feature_importances.items(), key=lambda x: x[1], reverse=True):
        print(f"    {feature}: {importance:.4f}")
    
    plot_top_features(initial_feature_importances, "Initial")

    # Changes in categories across perturbations
    for perturbation in np.linspace(0, 1, 11):
        perturbed_sample = sample.copy()
        for feature_idx in feature_importance:
            perturbed_sample[feature_idx] = (1 - perturbation) * sample[feature_idx]

        print(f"Perturbed values at perturbation {perturbation}: {perturbed_sample[feature_importance]}")

        perturbed_shap_values = explainer.shap_values(perturbed_sample)
        new_attack_cat_code = attack_cat_model.predict([perturbed_sample])[0]
        new_attack_cat = attack_categories[new_attack_cat_code]

        print(f"Perturbation {perturbation:.1f}: Attack category changes to {new_attack_cat} (Code: {new_attack_cat_code})")
        feature_importances = {feature_names[idx]: np.abs(perturbed_shap_values).mean(0)[idx] for idx in feature_importance}
        for feature, importance in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
            print(f"    {feature}: {importance:.4f}")

        plot_top_features(feature_importances, perturbation)

def plot_top_features(feature_importances, perturbation):
    features, importances = zip(*sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))
    plt.figure(figsize=(10, 5))
    plt.title(f'Feature Importances: {perturbation}')
    plt.bar(features, importances, color='skyblue')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


analyze_completeness(0)

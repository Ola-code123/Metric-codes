import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Load UNSW-NB15 dataset
train_data = pd.read_csv('...')
test_data = pd.read_csv('...')

# Remove 'id' in dataset
if 'id' in train_data.columns:
    train_data.drop('id', axis=1, inplace=True)
    test_data.drop('id', axis=1, inplace=True)

# Encoding
category_columns = ['proto', 'service', 'state']  


combined_data = pd.concat([train_data[category_columns], test_data[category_columns]])
encoder = LabelEncoder()
for column in category_columns:
    combined_data[column] = encoder.fit_transform(combined_data[column])
    train_data[column] = encoder.transform(train_data[column])
    test_data[column] = encoder.transform(test_data[column])


train_data['attack_cat'] = train_data['attack_cat'].fillna('Unknown').astype(str)
test_data['attack_cat'] = test_data['attack_cat'].fillna('Unknown').astype(str)

# Mapping for attack categories
attack_mapping = {
    'Normal': '0', 'Fuzzers': '1', 'Analysis': '2', 'Backdoors': '3', 'DoS': '4',
    'Exploits': '5', 'Generic': '6', 'Reconnaissance': '7', 'Shellcode': '8', 'Worms': '9'
}
reverse_attack_mapping = {v: k for k, v in attack_mapping.items()}


train_data['attack_cat'] = train_data['attack_cat'].map(attack_mapping).fillna('Unknown')
test_data['attack_cat'] = test_data['attack_cat'].map(attack_mapping).fillna('Unknown')


all_attack_cats = pd.concat([train_data['attack_cat'], test_data['attack_cat']]).drop_duplicates()
encoder.fit(all_attack_cats)  


y_training = encoder.transform(train_data['attack_cat'])
y_testing = encoder.transform(test_data['attack_cat'])


exclusion_list = ['attack_cat', 'Label'] if 'Label' in train_data.columns else ['attack_cat']

X_training = train_data.drop(exclusion_list, axis=1)
X_testing = test_data.drop(exclusion_list, axis=1)

# Scale features
scaler = StandardScaler()
X_training = scaler.fit_transform(X_training)
X_testing = scaler.transform(X_testing)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_training, y_training)

# Starting a LIME explainer
explainer = LimeTabularExplainer(X_training, feature_names=[col for col in train_data.columns if col not in exclusion_list],
                                 class_names=encoder.classes_, discretize_continuous=True)

# Function to analyze completness with LIME 
def analyze_completeness(sample_index, num_features=5):
    original_sample = X_testing[sample_index]
    original_class_index = model.predict(original_sample.reshape(1, -1))[0]
    original_class_code = encoder.classes_[original_class_index]
    original_class_name = reverse_attack_mapping[original_class_code]

    print(f"Original attack category: {original_class_name} (Code: {original_class_index})\n")

    explanation = explainer.explain_instance(original_sample, model.predict_proba, num_features=num_features)
    feature_importances = {feature: weight for feature, weight in explanation.as_list()}

    print("Top initial feature importances:")
    for feature, weight in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True):
        print(f"    {feature}: {weight:.4f}")

    perturbation_levels = np.linspace(0, 1, 11)
    for perturbation in perturbation_levels:
        perturbed_sample = original_sample.copy() * (1 - perturbation)
        perturbed_exp = explainer.explain_instance(perturbed_sample, model.predict_proba, num_features=num_features)
        perturbed_class_index = model.predict(perturbed_sample.reshape(1, -1))[0]
        perturbed_class_code = encoder.classes_[perturbed_class_index]
        perturbed_class_name = reverse_attack_mapping[perturbed_class_code]
        perturbed_feature_importances = {feature: weight for feature, weight in perturbed_exp.as_list()}

        print(f"\nPerturbation {perturbation:.1f}")
        print(f"New attack category: {perturbed_class_name} (Code: {perturbed_class_index})")
        if original_class_index == perturbed_class_index:
            print("No change in attack category")
        else:
            print("Change in attack category")
        for feature, weight in sorted(perturbed_feature_importances.items(), key=lambda item: item[1], reverse=True):
            print(f"    {feature}: {weight:.4f}")

        # Plotting the feature importances
        labels, values = zip(*sorted(perturbed_feature_importances.items(), key=lambda item: item[1], reverse=True))
        indexes = np.arange(len(labels))
        plt.figure(figsize=(10, 5))
        plt.bar(indexes, values, color='skyblue')
        plt.xticks(indexes, labels, rotation=45, ha='right')  
        plt.title(f'Feature Importances at Perturbation {perturbation:.1f}')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()  
        plt.show()


analyze_completeness(0)


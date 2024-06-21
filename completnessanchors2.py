import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from alibi.explainers import AnchorTabular

# Define the column names for the NSL-KDD dataset
col_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

# Load NSL-KDD dataset
train_data = pd.read_csv('...', names=col_names)
test_data = pd.read_csv('...', names=col_names)

# Encode 
encoder = LabelEncoder()
categorical_columns = ['protocol_type', 'service', 'flag']
for column in categorical_columns:
    combined = pd.concat([train_data[column], test_data[column]])  
    encoder.fit(combined)
    train_data[column] = encoder.transform(train_data[column])
    test_data[column] = encoder.transform(test_data[column])

# Mapping of attack categories
attack_mapping = {
    'normal': 'Normal', 'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R'
}

train_data['label'] = train_data['label'].map(attack_mapping).fillna('Unknown')
test_data['label'] = test_data['label'].map(attack_mapping).fillna('Unknown')


all_labels = pd.concat([train_data['label'], test_data['label']])
encoder.fit(all_labels.unique())


y_training = encoder.transform(train_data['label'])
y_testing = encoder.transform(test_data['label'])

feature_cols = [col for col in col_names if col not in ('label', 'num_outbound_cmds')]
X_training = train_data[feature_cols]
X_testing = test_data[feature_cols]

# Scaling features
scaler = StandardScaler()
X_training = scaler.fit_transform(X_training)
X_testing = scaler.transform(X_testing)

# Training model for predicting attack categories
attack_cat_model = RandomForestClassifier(n_estimators=100, random_state=42)
attack_cat_model.fit(X_training, y_training)

# Starting Anchor explainer
feature_names = feature_cols
explainer = AnchorTabular(predictor=attack_cat_model.predict, feature_names=feature_names)
explainer.fit(X_training)

def plot_feature_importances(features, values, perturbation):
    plt.figure(figsize=(10, 5))
    plt.bar(features, values, color='skyblue')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.title(f'Feature Values at Perturbation {perturbation:.1f}')
    plt.xticks(rotation=45, ha='right')  
    plt.tight_layout()
    plt.show()

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

def analyze_completeness(sample_index, threshold=0.75):
    sample = X_testing[sample_index]
    original_prediction = attack_cat_model.predict([sample])[0]
    original_attack_cat = encoder.inverse_transform([original_prediction])[0]
    print(f"Original attack category: {original_attack_cat} (Code: {original_prediction})\n")
    
    explanation = explainer.explain(sample, threshold=threshold)
    print("Initial Anchor:", explanation.anchor if explanation.anchor else "No anchor formed")
    print("Initial Precision: %.2f" % explanation.precision)
    print("Initial Coverage: %.2f" % explanation.coverage)
    
    perturbation_results = []
    
    if explanation.anchor:
        # Extracting feature names from the anchor conditions
        feature_indices = []
        for condition in explanation.anchor:
            feature_name = condition.split(' ')[0]  
            if feature_name in feature_names:
                feature_indices.append(feature_names.index(feature_name))
            else:
                print(f"Feature name '{feature_name}' from anchor '{condition}' not found in feature list.")
                continue

        for perturbation in np.linspace(0, 1, 11):
            perturbed_sample = np.array(sample, copy=True)
            current_values = []
            
            for feature_idx in feature_indices:
                original_value = perturbed_sample[feature_idx]
                perturbed_sample[feature_idx] = original_value * (1 - perturbation)
                current_values.append(perturbed_sample[feature_idx])
            
            perturbed_prediction = attack_cat_model.predict([perturbed_sample])[0]
            new_attack_cat = encoder.inverse_transform([perturbed_prediction])[0]
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

analyze_completeness(0, threshold=0.75)

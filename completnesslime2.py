import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

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
    train_data[column] = encoder.fit_transform(train_data[column])
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


encoder.fit(pd.concat([train_data['label'], test_data['label']]))
y_training = encoder.transform(train_data['label'])
y_testing = encoder.transform(test_data['label'])


feature_cols = [col for col in col_names if col not in ('label', 'num_outbound_cmds')]
X_training = train_data[feature_cols]
X_testing = test_data[feature_cols]

# Scale features
scaler = StandardScaler()
X_training = scaler.fit_transform(X_training)
X_testing = scaler.transform(X_testing)

# Training a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_training, y_training)

# Starting LIME explainer
explainer = LimeTabularExplainer(X_training, feature_names=feature_cols, class_names=encoder.classes_, discretize_continuous=True)

# Function to analyze completness with LIME 
def analyze_completness(sample_index, num_features=5):
    original_sample = X_testing[sample_index]
    original_class_index = model.predict(original_sample.reshape(1, -1))[0]
    original_class = encoder.classes_[original_class_index]
    print(f"Original attack category: {original_class} (Code: {original_class_index})\n")

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
        perturbed_class = encoder.classes_[perturbed_class_index]
        
        print(f"\nPerturbation {perturbation:.1f}")
        print(f"New attack category: {perturbed_class} (Code: {perturbed_class_index})")
        if original_class_index == perturbed_class_index:
            print("No change in attack category")
        else:
            print("Change in attack category")

        # Plotting the feature importances
        labels, values = zip(*sorted({feature: weight for feature, weight in perturbed_exp.as_list()}.items(), key=lambda item: item[1], reverse=True))
        indexes = np.arange(len(labels))
        plt.figure(figsize=(12, 8))  
        plt.bar(indexes, values, color='skyblue')
        plt.xticks(indexes, labels, rotation=45, ha='right')  
        plt.subplots_adjust(bottom=0.3)  
        plt.title(f'Feature Importances at Perturbation {perturbation:.1f}')
        plt.show()


analyze_completness(0)

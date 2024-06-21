import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap
import matplotlib.pyplot as plt

# Defining the column names for the NSL-KDD dataset
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

# Add paths in '...' to the NSL-KDD train and test dataset
train_data = pd.read_csv('...', names=col_names)
test_data = pd.read_csv('...', names=col_names)

 
category_columns = ['protocol_type', 'service', 'flag']

# Encoding with LabelEncoder
train_encoders = {}
test_encoders = {}
for column in category_columns:
    train_encoders[column] = LabelEncoder()
    test_encoders[column] = LabelEncoder()
    train_data[column] = train_encoders[column].fit_transform(train_data[column])
    test_data[column] = test_encoders[column].fit_transform(test_data[column])

# Attack mapping
attack_mapping = {
    'normal': 'Normal', 'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R'
}
train_data['label'] = train_data['label'].map(attack_mapping).fillna('Unknown')
test_data['label'] = test_data['label'].map(attack_mapping).fillna('Unknown')

# Encoding
label_encoder = LabelEncoder()
label_encoder.fit(np.unique(train_data['label'].tolist() + test_data['label'].tolist()))
y_training = label_encoder.transform(train_data['label'])
y_testing = label_encoder.transform(test_data['label'])

# Prepare datasets
feature_cols = [col for col in col_names if col not in ('label', 'num_outbound_cmds')]
X_training = train_data[feature_cols]
X_testing = test_data[feature_cols]

# Scaling features
scaler = StandardScaler()
X_training = scaler.fit_transform(X_training)
X_testing = scaler.transform(X_testing)

# Training the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_training, y_training)

# Initialize SHAP Explainer
explainer = shap.TreeExplainer(model)
sample_X_testing = X_testing[np.random.choice(X_testing.shape[0], 300, replace=False)]
shap_values = explainer.shap_values(sample_X_testing)
shap.summary_plot(shap_values, sample_X_testing, plot_type="bar")

# Function to plot feature importances
def plot_top_features(feature_importances, perturbation, title='Feature Importances'):
    features, importances = zip(*sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))
    plt.figure(figsize=(10, 5))
    plt.bar(features, importances, color='skyblue')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'{title} - {perturbation}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Function to analyze how feature importance changes with perturbation
def analyze_completeness(sample_index, top_n_features=5):
    # Generate a single sample and its original attack category
    sample = X_testing[sample_index]
    original_attack_cat_code = model.predict([sample])[0]
    original_attack_cat = label_encoder.inverse_transform([original_attack_cat_code])[0]
    original_shap_values = explainer.shap_values(sample)

    # Identify top features based on SHAP values
    feature_importance = np.argsort(np.abs(original_shap_values).mean(0))[::-1][:top_n_features]
    initial_feature_importances = {feature_cols[idx]: np.abs(original_shap_values).mean(0)[idx] for idx in feature_importance}

    print(f"Original attack category: {original_attack_cat} (Code: {original_attack_cat_code})\n")
    print("Top initial feature importances:")
    for feature, importance in sorted(initial_feature_importances.items(), key=lambda x: x[1], reverse=True):
        print(f"    {feature}: {importance:.4f}")

    # Plot initial importances
    plot_top_features(initial_feature_importances, 'Initial', 'Initial Feature Importances')

    # Perturb features and analyze impact
    for perturbation in np.linspace(0, 1, 11):
        perturbed_sample = sample.copy()
        for feature_idx in feature_importance:
            perturbed_sample[feature_idx] *= (1 - perturbation)

        perturbed_shap_values = explainer.shap_values(perturbed_sample)
        new_attack_cat_code = model.predict([perturbed_sample])[0]
        new_attack_cat = label_encoder.inverse_transform([new_attack_cat_code])[0]

        # Recalculate and show feature importances after perturbation
        feature_importances = {feature_cols[idx]: np.abs(perturbed_shap_values).mean(0)[idx] for idx in feature_importance}
        print(f"Perturbation {perturbation:.1f}: Attack category changes to {new_attack_cat} (Code: {new_attack_cat_code})")
        for feature, importance in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
            print(f"    {feature}: {importance:.4f}")

        # Plot importances for each perturbation
        plot_top_features(feature_importances, perturbation)

# Analyzing first sample
analyze_completeness(0)

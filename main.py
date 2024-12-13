import pandas as pd
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import tldextract
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler

def extract_tld(url):
    extracted = tldextract.extract(url)
    return extracted.suffix if extracted.suffix else 'unknown'


def calculate_special_char_ratio(url):
    special_chars = len(re.findall(r'[@\-_?|=$%!]', url))
    total_chars = len(url)
    normal_chars = total_chars - special_chars
    
    if normal_chars == 0:  # Avoid division by zero
        return 1.0  # If all characters are special, return 1
    
    return special_chars / normal_chars

# Load the training dataset
data = pd.read_csv("combined_noHttp.csv")

# Feature engineering for training data
data['url_length'] = data['URL'].apply(len)
data['special_char_count'] = data['URL'].apply(lambda x: len(re.findall(r'[@\-_?|=$%!]', x)))
data['special_char_ratio'] = data['URL'].apply(calculate_special_char_ratio)
data['subdomain_count'] = data['URL'].apply(lambda x: x.count('.'))

suspicious_keywords = ['free', 'verify', 'bank', 'secure', 'account', 'update', 'coin', 'crypto', 'validate', 'reward']
data['suspicious_keyword'] = data['URL'].apply(lambda x: any(keyword in x for keyword in suspicious_keywords))

data['tld'] = data['URL'].apply(extract_tld)

# Load the testing dataset
test_data = pd.read_csv("paperTest.csv")
test_data['URL'] = test_data['URL'].fillna('')
test_data['URL'] = test_data['URL'].astype(str)

# Feature engineering for testing data
test_data['url_length'] = test_data['URL'].apply(len)
test_data['special_char_count'] = test_data['URL'].apply(lambda x: len(re.findall(r'[@\-_?|=$%!]', x)))
test_data['special_char_ratio'] = test_data['URL'].apply(calculate_special_char_ratio)
test_data['subdomain_count'] = test_data['URL'].apply(lambda x: x.count('.'))
test_data['suspicious_keyword'] = test_data['URL'].apply(lambda x: any(keyword in x for keyword in suspicious_keywords))
test_data['tld'] = test_data['URL'].apply(extract_tld)

le = LabelEncoder()
data['tld_encoded'] = le.fit_transform(data['tld'])

def safe_transform(x):
    try:
        return le.transform([x])[0]
    except ValueError:
        return -1
    
test_data['tld_encoded'] = test_data['tld'].apply(safe_transform)

# Identify overlapping URLs
overlap = set(data['URL']).intersection(set(test_data['URL']))

# Remove overlapping URLs from the test dataset
test_data = test_data[~test_data['URL'].isin(overlap)]

# Print the number of remaining rows in the test dataset
print(f"Number of rows in the test dataset after removing overlap: {len(test_data)}")

# Prepare training and testing data
X_train = data[['url_length', 'special_char_count','special_char_ratio', 'subdomain_count', 'suspicious_keyword', 'tld_encoded']]
y_train = data['label']

X_test = test_data[['url_length', 'special_char_count','special_char_ratio', 'subdomain_count', 'suspicious_keyword', 'tld_encoded']]
y_test = test_data['label']

# Standardize the features for ANN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the models
nb_model = GaussianNB()
dt_model = DecisionTreeClassifier(random_state=42)
ann_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

#Train
models = {
    "Naive Bayes": (nb_model, X_train, X_test),
    "Decision Tree": (dt_model, X_train, X_test),
    "ANN": (ann_model, X_train_scaled, X_test_scaled)
}

results = {}

for name, (model, X_train_model, X_test_model) in models.items():
    model.fit(X_train_model, y_train)
    y_pred = model.predict(X_test_model)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    results[name] = {
        "Accuracy": accuracy,
        "Precision_0": precision[0],
        "Precision_1": precision[1],
        "Recall_0": recall[0],
        "Recall_1": recall[1],
        "F1-Score_0": f1[0],
        "F1-Score_1": f1[1]
    }
    
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Create comparison chart
metrics = ["Accuracy", "Precision_0", "Precision_1", "Recall_0", "Recall_1", "F1-Score_0", "F1-Score_1"]
model_names = list(results.keys())

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(15, 8))

for i, model_name in enumerate(model_names):
    values = [results[model_name][metric] for metric in metrics]
    ax.bar(x + i*width, values, width, label=model_name)

ax.set_ylabel('Scores')
ax.set_title('Model Comparison')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, rotation=45, ha='right')
ax.legend()

# Set y-axis to range from 0 to 1
ax.set_ylim(0, 1)

# Add value labels on top of each bar
for i, model_name in enumerate(model_names):
    values = [results[model_name][metric] for metric in metrics]
    for j, v in enumerate(values):
        ax.text(j + i*width, v, f'{v:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()



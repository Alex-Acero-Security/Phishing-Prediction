import pandas as pd

# Load datasets
uci_data = pd.read_excel("UCI train.xlsx")
phishStats_data = pd.read_excel("phishStats.xlsx")
tranco_data = pd.read_excel("tranco_dataset_fixed.xlsx")

# Combine legitimate and phishing entries
all_data = pd.concat([uci_data, phishStats_data, tranco_data])

# Remove duplicates
all_data = all_data.drop_duplicates(subset=['URL'])

# Save to a single file
all_data.to_csv("combined_dataset.csv", index=False)
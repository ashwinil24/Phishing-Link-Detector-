import pandas as pd

# Load datasets
df_phishing = pd.read_csv("D:/phishing project/phishing_dataset.csv")  # 501 phishing links
df_legit = pd.read_csv("D:/phishing project/legit_links.csv", header=None, names=["id", "url"])  # 100k legit links

# Drop 'id' column from legit dataset
df_legit.drop(columns=["id"], inplace=True)

# Add labels
df_phishing["label"] = "phishing"
df_legit["label"] = "legit"

# Downsample legit links to 500
df_legit_sampled = df_legit.sample(n=500, random_state=42)  # Ensures reproducibility

# Merge datasets
df_combined = pd.concat([df_phishing, df_legit_sampled], ignore_index=True)

# Save the balanced dataset
df_combined.to_csv("D:/phishing project/balanced_dataset.csv", index=False)

print("✅ Dataset balanced and saved as balanced_dataset.csv")

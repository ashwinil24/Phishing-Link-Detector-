import pandas as pd  

# Load the phishing dataset  
df_phishing = pd.read_csv(r"D:\phishing project\phishing_dataset.csv")  

# Drop 'id' column if present in phishing dataset
if "id" in df_phishing.columns:
    df_phishing.drop(columns=["id"], inplace=True)

# Load the legitimate dataset  
df_legit = pd.read_csv(r"D:\phishing project\legit_links.csv", header=None, names=["id", "url"])  

# Drop the 'id' column from legitimate dataset  
df_legit.drop(columns=["id"], inplace=True)  

# Add label columns  
df_phishing["label"] = "phishing"  
df_legit["label"] = "legit"  

# Downsample the legitimate dataset to match phishing dataset count (500 phishing samples)
df_legit_sampled = df_legit.sample(n=500, random_state=42)

# Downsample the phishing dataset to 500 if needed
df_phishing_sampled = df_phishing.sample(n=500, random_state=42) if len(df_phishing) > 500 else df_phishing

# Merge both balanced datasets  
df_combined = pd.concat([df_phishing_sampled, df_legit_sampled], ignore_index=True)  

# Save the balanced dataset  
df_combined.to_csv(r"D:\phishing project\balanced_dataset.csv", index=False)  

# Display first few rows to verify  
print("✅ Balanced dataset created successfully! Here's a preview:")  
print(df_combined.head())
print("📌 Final Columns:", df_combined.columns)
print(f"📊 Final Label Distribution:\n{df_combined['label'].value_counts()}")

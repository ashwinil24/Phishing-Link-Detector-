import pandas as pd
import os

# Define file paths
input_file = os.path.join("D:", "phishing project", "phishing_dataset.csv")
output_file = os.path.join("D:", "phishing project", "phishing_dataset_cleaned.csv")

# Load dataset
df = pd.read_csv("phishing_dataset.csv")

# Track initial shape
initial_rows = df.shape[0]

# Drop duplicates
df.drop_duplicates(inplace=True)
duplicates_removed = initial_rows - df.shape[0]

# Drop rows with missing values
df.dropna(inplace=True)
nan_removed = initial_rows - duplicates_removed - df.shape[0]

# Reset index
df.reset_index(drop=True, inplace=True)

# Save cleaned dataset
df.to_csv("phishing_dataset_cleaned.csv", index=False)

# Print summary
print(f"✅ Dataset cleaned and saved as: {output_file}")
print(f"📊 Cleaning Summary:")
print(f"   🔹 Initial Rows: {initial_rows}")
print(f"   🔹 Duplicates Removed: {duplicates_removed}")
print(f"   🔹 NaN Rows Removed: {nan_removed}")
print(f"   🔹 Final Rows: {df.shape[0]}")



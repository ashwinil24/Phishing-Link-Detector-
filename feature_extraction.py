import pandas as pd
from urllib.parse import urlparse
import re

# Load the dataset
file_path = "D:/phishing project/phishing_dataset_cleaned.csv"
df = pd.read_csv(file_path)

# Ensure 'url' column exists
if "url" not in df.columns:
    raise ValueError("❌ Error: 'url' column not found in dataset!")

# List of suspicious keywords commonly found in phishing URLs
suspicious_keywords = {"secure", "account", "login", "bank", "verify", "update", "password", "free", "offer", "win", "prize"}

# List of common URL shortening services
shorteners = {'bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly', 'is.gd', 'buff.ly', 'adf.ly'}

# Function to extract features from a URL
def extract_features(url):
    if not isinstance(url, str) or url.strip() == "":
        return {
            "url_length": 0, "dot_count": 0, "slash_count": 0, "subdomain_count": 0,
            "suspicious_keyword": 0, "special_char_count": 0, "digit_count": 0, 
            "url_encoding_count": 0, "has_ip": 0, "is_shortened": 0, 
            "repeated_char_count": 0, "https_present": 0, "has_at_symbol": 0
        }  

    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    return {
        "url_length": len(url),
        "dot_count": url.count('.'),
        "slash_count": url.count('/'),
        "subdomain_count": domain.count('.'),
        "suspicious_keyword": int(any(keyword in url.lower() for keyword in suspicious_keywords)),
        "special_char_count": len(re.findall(r'[-_@=?.%#]', url)),
        "digit_count": len(re.findall(r'\d', url)),
        "url_encoding_count": url.count('%'),
        "has_ip": int(bool(re.match(r'^(?:\d{1,3}\.){3}\d{1,3}$', domain))),
        "is_shortened": int(any(shortener in url for shortener in shorteners)),
        "repeated_char_count": len(re.findall(r'(.)\1{2,}', url)),
        "https_present": int(url.startswith("https")),
        "has_at_symbol": int("@" in url)  # New feature added
    }
df_features = pd.DataFrame.from_records(df["url"].apply(extract_features))

# Merge extracted features with the original dataset
df_final = pd.concat([df, df_features], axis=1)

# Save updated dataset
output_path = "D:/phishing project/phishing_dataset_with_features.csv"
df_final.to_csv(output_path, index=False)



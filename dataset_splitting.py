import pandas as pd  
import re  
from urllib.parse import urlparse  
from sklearn.model_selection import train_test_split  

# Load phishing dataset  
df_phishing = pd.read_csv("phishing_dataset_with_features.csv")  
df_phishing["label"] = "phishing"  

# Load legitimate dataset  
df_legit = pd.read_csv("legit_links.csv", names=["url"])  
df_legit["label"] = "legit"  

# Drop missing values  
df_legit = df_legit.dropna(subset=["url"])  

# 🔹 Function to check if URL contains suspicious keywords
def contains_suspicious_keyword(url):
    suspicious_words = ["login", "verify", "update", "bank", "secure", "account", 
                        "password", "confirm", "security", "billing", "webscr"]  
    return any(word in url.lower() for word in suspicious_words)

# 🔹 Function to check if a URL contains an IP address
def contains_ip(url):
    ip_pattern = r'^(http[s]?://)?(\d{1,3}\.){3}\d{1,3}(/|$)'  # Matches IPv4 addresses
    return bool(re.match(ip_pattern, url))

# 🔹 Function to check if a URL is from a known URL shortener
def is_shortened(url):
    shortened_domains = ["bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly", "is.gd", 
                         "buff.ly", "adf.ly", "shorturl.at"]
    parsed_url = urlparse(url).netloc
    return parsed_url in shortened_domains

# Feature extraction for legitimate dataset  
df_legit["url_length"] = df_legit["url"].apply(len)  
df_legit["dot_count"] = df_legit["url"].apply(lambda x: x.count('.'))  
df_legit["slash_count"] = df_legit["url"].apply(lambda x: x.count('/'))  
df_legit["subdomain_count"] = df_legit["url"].apply(lambda x: x.count('.') - 1)  
df_legit["suspicious_keyword"] = df_legit["url"].apply(contains_suspicious_keyword).astype(int)  
df_legit["special_char_count"] = df_legit["url"].apply(lambda x: sum(not c.isalnum() for c in x))  
df_legit["has_at_symbol"] = df_legit["url"].apply(lambda x: '@' in x).astype(int)  
df_legit["digit_count"] = df_legit["url"].apply(lambda x: sum(c.isdigit() for c in x))  
df_legit["https_present"] = df_legit["url"].apply(lambda x: x.startswith("https")).astype(int)  
df_legit["url_encoding_count"] = df_legit["url"].apply(lambda x: x.count('%'))  
df_legit["has_ip"] = df_legit["url"].apply(contains_ip).astype(int)  
df_legit["is_shortened"] = df_legit["url"].apply(is_shortened).astype(int)  
df_legit["repeated_char_count"] = df_legit["url"].apply(lambda x: sum(x.count(c) > 2 for c in set(x)))  

# Drop raw URL column  
df_legit.drop(columns=["url"], inplace=True)  

# Balance the dataset  
phishing_count = df_phishing.shape[0]  
df_legit_sampled = df_legit.sample(n=phishing_count, random_state=42) if len(df_legit) >= phishing_count else df_legit  

# Merge phishing and legitimate datasets  
df_balanced = pd.concat([df_phishing, df_legit_sampled], ignore_index=True)  

# Define features  
feature_columns = ["url_length", "dot_count", "slash_count", "subdomain_count",  
                   "suspicious_keyword", "special_char_count", "has_at_symbol",  
                   "digit_count", "https_present", "url_encoding_count", "has_ip",  
                   "is_shortened", "repeated_char_count"]  

# Ensure all expected features exist  
missing_features = [col for col in feature_columns if col not in df_balanced.columns]  
if missing_features:  
    raise ValueError(f"❌ Missing features in dataset: {missing_features}")  

# Define X (features) and y (labels)  
X = df_balanced[feature_columns]  
y = df_balanced["label"].map({"phishing": 1, "legit": 0})  

# Print class distribution after balancing  
print(f"📌 Balanced Class Distribution: {y.value_counts().to_dict()}")  

# Split data into 80% training and 20% testing (Stratified)  
X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size=0.20, random_state=42, stratify=y  
)  

# Print class distribution after splitting  
print(f"✅ Training Distribution: {y_train.value_counts().to_dict()}")  
print(f"✅ Testing Distribution: {y_test.value_counts().to_dict()}")  
print("✅ Data split completed: 80% training, 20% testing with balanced classes.")  
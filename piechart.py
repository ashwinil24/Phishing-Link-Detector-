import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.metrics import confusion_matrix  
import pandas as pd  
import numpy as np  
import joblib  
from dataset_splitting import X_test, y_test  

# Load trained model  
model_filename = "rf_model_v4.pkl"  # Ensure this matches your saved model  
rf_model = joblib.load(model_filename)  

# Make predictions  
y_pred = rf_model.predict(X_test)  

# Compute confusion matrix  
cm = confusion_matrix(y_test, y_pred)  
labels = ["Legit", "Phishing"]  

# 🔹 Plot Confusion Matrix  
plt.figure(figsize=(6, 5))  
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)  
plt.xlabel("Predicted")  
plt.ylabel("Actual")  
plt.title("Confusion Matrix")  
plt.show()  

# 🔹 Plot Class Distribution as a Pie Chart  
unique, counts = np.unique(y_pred, return_counts=True)  
class_distribution = dict(zip(["Legit", "Phishing"], counts))  

plt.figure(figsize=(6, 6))  
plt.pie(counts, labels=class_distribution.keys(), autopct="%1.1f%%", colors=["green", "red"], startangle=90)  
plt.title("Class Distribution of Predictions")  
plt.show()  
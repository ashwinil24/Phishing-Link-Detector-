from dataset_splitting import X_train, y_train, X_test, y_test
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Check unique classes in y_train
unique_classes = set(y_train)
print(f"🔍 Unique Classes in Training Data: {unique_classes}")

# Ensure binary classification
if len(unique_classes) != 2:
    raise ValueError("❌ Training data does not contain exactly two classes!")

# Initialize Random Forest model with class balancing
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Perform 5-fold cross-validation to estimate performance
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"📊 Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train the model on the training data
rf_model.fit(X_train, y_train)
print("✅ Random Forest model training completed.")

# Evaluate on test set
y_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"📌 Test Accuracy: {test_accuracy:.4f}")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Check feature importance
feature_importance = rf_model.feature_importances_
feature_importance_dict = dict(zip(X_train.columns, feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("\n📌 Top 5 Important Features:")
for feature, importance in sorted_features[:5]:
    print(f"🔹 {feature}: {importance:.4f}")

# Save model with versioning
version = 1
model_filename = f"rf_model_v{4}.pkl"
while os.path.exists(model_filename):
    version += 1
    model_filename = f"rf_model_v{4}.pkl"

joblib.dump(rf_model, model_filename)
print(f"✅ Random Forest model trained and saved as {model_filename}")
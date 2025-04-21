import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Features (X) and target (y)
X = data.drop('target', axis=1)  # Drop the 'target' column to get features
y = data['target']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test = scaler.transform(X_test)  # Transform the test data using the same scaler

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)  # Predict on the test data
accuracy = accuracy_score(y_test, y_pred)  # Compare predictions with actual values
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model and scaler
joblib.dump(model, 'iris_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
# Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Print a preview of the dataset (optional)
print("Dataset Preview:")
print(data.head())

# Split the data into features (X) and target (y)
X = data.iloc[:, :-1]  # Features: Sepal and petal dimensions
y = data['species']    # Target: Flower species

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # Train the model

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example predictions
sample_data = [[5.1, 3.5, 1.4, 0.2],  # Example 1
               [6.2, 3.4, 5.4, 2.3]]  # Example 2
sample_prediction = model.predict(sample_data)
print("\nPredicted species for the sample data:", sample_prediction)

# Mapping numeric predictions to species names
species_names = iris.target_names
sample_species = [species_names[pred] for pred in sample_prediction]
print("\nSample Data Predictions:")
for i, species in enumerate(sample_species):
    print(f"Sample {i+1}: {species}")

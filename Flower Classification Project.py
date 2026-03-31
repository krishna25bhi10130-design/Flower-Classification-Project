# ------------------------------
# Iris Flower Classification Project (AIML)
# ------------------------------

# 1. Import Libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 2. Load Dataset
iris = load_iris()
X = iris.data                    # features
y = iris.target                  # labels

# 3. Convert into DataFrame for easy viewing
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
print("Dataset Preview:")
print(df.head())

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 5. Model Training
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 6. Prediction
y_pred = model.predict(X_test)

# 7. Accuracy
acc = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", acc)

# 8. Predict New Sample
sample = [[5.1, 3.5, 1.4, 0.2]]    # Example sepal/petal values
prediction = model.predict(sample)

print("\nPredicted Flower Type:", iris.target_names[prediction][0])
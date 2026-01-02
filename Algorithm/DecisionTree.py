import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# 1. Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and train the Decision Tree
# We set 'max_depth' to keep the tree from getting too complex (overfitting)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 4. Make predictions
y_pred = model.predict(X_test)

# 5. Evaluate
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# 6. Visualize the Tree
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()
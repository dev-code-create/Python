from sklearn.tree import DecisionTreeClassifier

X = [[25], [30], [35], [40]]
y = [0, 0, 1, 1]

model = DecisionTreeClassifier
model.fit(X, y)

model.predict([[32]])
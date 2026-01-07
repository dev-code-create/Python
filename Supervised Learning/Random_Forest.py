from sklearn.ensemble import RandomForestClassifier

X = [[25], [30], [35], [40]]
y = [0, 0, 1, 1]

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X, y)
model.predict([[32]])
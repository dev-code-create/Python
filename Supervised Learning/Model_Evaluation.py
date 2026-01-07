from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

X = [[25], [30], [35], [40]]
y = [0, 0, 1, 1]

model = RandomForestClassifier()
model.fit(X,y)

y_pred = model.predict(X)
accuracy_score(y,y_pred)
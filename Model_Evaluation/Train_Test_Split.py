from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

model = LogisticRegression()

X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model.fit(X_train, y_train)

scores = cross_val_score(model,X,y,cv=3)
print(scores)
print(scores.mean())

y_pred = model.predict(X_test)
confusion_matrix(y_test,y_pred)

model.fit(X_train, y_train)


accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test, y_pred)

print(precision)
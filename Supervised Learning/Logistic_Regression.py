#Logistic gives probabiliy (0 -1) (yes or no) ex-> spam detection , student pass or fail
#sigmoid functionn 1/(1+e^-x)

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

X = [[20], [25], [30], [35]]
y = [0, 0, 1, 1]  # not buy / buy

model = LogisticRegression()
model.fit(X,y)

model.predict([[28]])
model.predict_proba([[28]])

print("Prediction:", model.predict([[28]]))
print("Probability:", model.predict_proba([[28]]))


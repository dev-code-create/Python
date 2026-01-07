#Predict continous values ex- House price , salary , temp
# y = mx + b (m = slope , b = intercept)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

X = np.array([[500],[1000],[1500],[2000]])
y = np.array([20,40,60,80])

model = LinearRegression()
model.fit(X,y)

print(model.predict([[1200]]))


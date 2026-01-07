#Price depends on multiple factors

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

X = pd.DataFrame({
  "Area":[1000,1500,2000],
  "Bedrooms":[2,3,4]
})

y = [50,70,90]

model = LinearRegression()
model.fit(X,y)

model.predict([[1200,3]])
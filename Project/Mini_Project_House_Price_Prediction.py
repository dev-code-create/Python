import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV

data = {
    "Area": [
        600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050,
        1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550,
        1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050,
        2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550,
        2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000, 3050
    ],

    "Bedrooms": [
        1,1,1,2,2,2,2,2,3,3,
        3,3,3,3,3,3,3,3,3,3,
        4,4,4,4,4,4,4,4,4,4,
        4,4,4,4,4,4,4,4,4,4,
        5,5,5,5,5,5,5,5,5,5
    ],

    "Bathrooms": [
        1,1,1,1,1,2,2,2,2,2,
        2,2,2,2,2,2,2,2,2,2,
        3,3,3,3,3,3,3,3,3,3,
        3,3,3,3,3,3,3,3,3,3,
        4,4,4,4,4,4,4,4,4,4
    ],

    "Location": [
        "City","City","Suburb","Suburb","City","City","Suburb","Suburb","City","City",
        "Suburb","Suburb","City","City","Suburb","Suburb","City","City","Suburb","Suburb",
        "City","City","Suburb","Suburb","City","City","Suburb","Suburb","City","City",
        "Suburb","Suburb","City","City","Suburb","Suburb","City","City","Suburb","Suburb",
        "City","City","Suburb","Suburb","City","City","Suburb","Suburb","City","City"
    ],

    "Price": [
        28,30,32,35,38,42,45,48,52,55,
        58,60,62,65,68,70,72,75,78,80,
        82,85,88,90,92,95,98,100,102,105,
        108,110,112,115,118,120,122,125,128,130,
        135,138,140,145,148,150,155,158,160,165
    ]
}


df = pd.DataFrame(data)

df.info()
df.describe()

sns.scatterplot(x="Area", y="Price",data =df)
#plt.show()

df.isnull().sum()

df = pd.get_dummies(df,columns=["Location"],drop_first=True)
print(df)

X = df.drop("Price" , axis = 1)
y = df["Price"]

X_train , X_test , y_train , y_test = train_test_split(
  X,y,test_size=0.2,random_state=42
)

scaler = StandardScaler();

X_trained_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_trained_scaled , y_train)

y_pred_lr = lr.predict(X_test_scaled)

print("MAE:",mean_absolute_error(y_test,y_pred_lr))
print("RMSE:", np.sqrt(mean_absolute_error(y_test,y_pred_lr)))
print("R2:", r2_score(y_test, y_pred_lr))

rf = RandomForestRegressor(
  n_estimators=100,
  random_state=42,
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R2:", r2_score(y_test, y_pred_rf))

#Random Forest > Linear Regression

para_grid = {
  "n_estimators": [50,100],
  "max_depth":[None,5,10]
}

grid = GridSearchCV(
  RandomForestRegressor(random_state=42),
  para_grid,
  cv=2,
  scoring='r2'
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
grid.best_params_

#grid.best_estimator_	The best trained model
#grid.best_params_      The parameter values of that model

new_house = pd.DataFrame({
    "Area": [1300],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "Location_Suburb": [0]
})

predicted_price = best_model.predict(new_house)
print("Predicted Price (Lakhs):", predicted_price[0])


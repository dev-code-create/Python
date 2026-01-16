import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

data = {
    "Area": [800, 900, 1000, 1100, 1200, 1500, 1800],
    "Bedrooms": [1, 2, 2, 3, 3, 4, 4],
    "Bathrooms": [1, 1, 2, 2, 2, 3, 3],
    "Location": ["City", "City", "Suburb", "Suburb", "City", "City", "Suburb"],
    "Price": [40, 50, 55, 65, 70, 90, 100]  # in lakhs
}

df = pd.DataFrame(data)

df.info()
df.describe()

sns.scatterplot(x="Area", y="Price",data =df)
#plt.show()

df.isnull().sum()

df = pd.get_dummies(df,columns=["Location"],drop_first=True)
print(df)

X = df.drop("Prices" , axis = 1)
y = df["Prices"]

X_train , X_test , y_train , y_test = train_test_split(
  X,y,test_size=0.2,random_state=42
)
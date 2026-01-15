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
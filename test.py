import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

df = pd.read_csv("CarPrice_Assignment.csv")

df["doornumber"] = df["doornumber"].map({"two": 2, "four": 4})
df["cylindernumber"] = df["cylindernumber"].map({"four": 4, "six": 6, "five": 5, "three": 3,"twelve":12, "two":2,"eight":8})
df["enginelocation"] = df["enginelocation"].map({"front": 0, "rear": 1})
df["fueltype"] = df["fueltype"].map({"gas": 0, "diesel": 1})
df["aspiration"] = df["aspiration"].map({"std": 0, "turbo": 1})

df = df.drop(["CarName", "carbody", "drivewheel", "enginetype", "fuelsystem"], axis=1)

def correlation_for_dropping(df, threshhold):
    columns_to_drop = set()
    corr = df.corr()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i,j]) > threshhold:
                columns_to_drop.add(corr.columns[i])
    return columns_to_drop

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X = X.drop(correlation_for_dropping(X, 0.93), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=15)

scaler = StandardScaler()
linear = LinearRegression()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear.fit(X_train, y_train)

y_pred = linear.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"mae: {mae}")
print(f"mse: {mse}")
print(f"r2 score: {r2}")

print(df["price"].mean())



plt.scatter(y_test, y_pred)
plt.show()






























































































































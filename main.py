import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoCV, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

df = pd.read_csv("6-bank_customers.csv")

X = df.drop("subscribed", axis=1)
y = df["subscribed"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
logistic = LogisticRegression()

logistic.fit(X_train, y_train) 

y_pred = logistic.predict(X_test)

score = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred)

matrix = confusion_matrix(y_test, y_pred)

model = LogisticRegression()


penalty_values = ["l1","l2","elasticnet"]
c_values = [100,10,1,0.1,0.01]
solver = ["newton-cg","lbfgs","liblinear","sag","saga","newton-cholesky"]


params = dict(penalty = penalty_values, C = c_values, solver = solver)

print(params)


cv = StratifiedKFold()

#grid = GridSearchCV(estimator=model, param_grid=params, cv=cv, scoring="accuracy", n_jobs=-1)
#grid.fit(X_train, y_train)
#y_pred = grid.predict(X_test)



randomcv = RandomizedSearchCV(estimator=model, param_distributions=params, cv=5, n_iter=10, scoring="accuracy")
randomcv.fit(X_train, y_train)
print(randomcv.best_params_)

















































































































































































































































# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE

# TITANIC DATASET:

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("titanic_dataset.csv")

df

df.isnull().sum()

df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])

df['Embarked'] = le.fit_transform(df['Embarked'].astype(str))

imputer = SimpleImputer(missing_values=np.nan, strategy='median')

df[['Age']] = imputer.fit_transform(df[['Age']])

print("Feature selection")

X = df.iloc[:, :-1]

y = df.iloc[:, -1]

selector = SelectKBest(chi2, k=3)

X_new = selector.fit_transform(X, y)

print(X_new)

df_new = pd.DataFrame(X_new, columns=['Pclass', 'Age', 'Fare'])

df_new['Survived'] = y.values

df_new.to_csv('titanic_transformed.csv', index=False)

print(df_new)

# CARPRICE DATASET:

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.ensemble import ExtraTreesRegressor

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("CarPrice.csv")

df

df = df.drop(['car_ID', 'CarName'], axis=1)

le = LabelEncoder()

df['fueltype'] = le.fit_transform(df['fueltype'])

df['aspiration'] = le.fit_transform(df['aspiration'])

df['doornumber'] = le.fit_transform(df['doornumber'])

df['carbody'] = le.fit_transform(df['carbody'])

df['drivewheel'] = le.fit_transform(df['drivewheel'])

df['enginelocation'] = le.fit_transform(df['enginelocation'])

df['enginetype'] = le.fit_transform(df['enginetype'])

df['cylindernumber'] = le.fit_transform(df['cylindernumber'])

df['fuelsystem'] = le.fit_transform(df['fuelsystem'])

X = df.iloc[:, :-1]

y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("Univariate Selection")

selector = SelectKBest(score_func=f_regression, k=10)

X_train_new = selector.fit_transform(X_train, y_train)

mask = selector.get_support()

selected_features = X_train.columns[mask]

model = ExtraTreesRegressor()

model.fit(X_train, y_train)

importance = model.feature_importances_

indices = np.argsort(importance)[::-1]

selected_features = X_train.columns[indices][:10]

df_new = pd.concat([X_train[selected_features], y_train], axis=1)

df_new.to_csv('CarPrice_new.csv', index=False)

print(df_new)

# OUTPUT:

# TITANIC DATA:
![Screenshot (175)](https://github.com/Dhivya-bharathi88/EX.NO.7/assets/128019999/f3ce6c8a-4eea-4694-a317-15d3c4e7e9c6)
![Screenshot (176)](https://github.com/Dhivya-bharathi88/EX.NO.7/assets/128019999/3b265177-b9e9-4550-a5ce-a27befb7eb29)
![Screenshot (177)](https://github.com/Dhivya-bharathi88/EX.NO.7/assets/128019999/dca9bfa7-fa65-496e-93e2-ab40441fa327)
![Screenshot (178)](https://github.com/Dhivya-bharathi88/EX.NO.7/assets/128019999/b8daf7a6-d97d-4674-9bbb-9d5f337a7f87)
# CARPRICE DATASET:
![Screenshot (179)](https://github.com/Dhivya-bharathi88/EX.NO.7/assets/128019999/7d045aa8-5161-4968-b456-22ea8c348c8d)
![Screenshot (180)](https://github.com/Dhivya-bharathi88/EX.NO.7/assets/128019999/31eaec81-fe3c-4e64-b57f-a60a7e7008b6)

# RESULT:
Thus the various feature selection techniques was performed on the given datasets and output was verified successfully.


# -*- coding: utf-8 -*-
"""Diabete Multiple Linear Regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1v090G_LH9yUuvnDvqWvTOxAN6a0iHtMU

# 1) Data Preprocessing
"""

from google.colab import drive
drive.mount('/content/drive')

"""## Importing the libraries"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv("/content/drive/MyDrive/diabetes.csv")

dataset.head()

dataset.columns

dataset.count()

dataset.describe()

dataset.info()

x = dataset.iloc[: , 0:-1].values

print(x)

y = dataset.iloc[: , -1].values

print(y)

"""## Encoding categorical data

## Splitting the dataset
"""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

len(x_train)

len(x_test)

"""# 2) Training the model"""

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(x_train, y_train)

classifier.intercept_

classifier.coef_

"""# 3) Predicting the Test set results"""

y_pred = classifier.predict(x_test)

y_pred

y_test

df = pd.DataFrame()
df["y_actual"] = y_test 
df["y_pred"] = y_pred
print(df)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ACC_S = accuracy_score(y_test, y_pred)
cm

ACC_S


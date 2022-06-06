#Autor; Hossein ZOLFAGHARI
#France
#2022
#Diabete Multiple Linear Regression.py

# 1) Data Preprocessing

"""## Importing the libraries"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv("diabetes.csv")
dataset.head()

x = dataset.iloc[: , 0:-1].values
#print(x)

y = dataset.iloc[: , -1].values
#print(y)


## Splitting the dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# 2) Training the model

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(x_train, y_train)


# 3) Predicting the Test set results

y_pred = classifier.predict(x_test)
df = pd.DataFrame()
df["y_actual"] = y_test 
df["y_pred"] = y_pred
print(df)


#Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ACC_S = accuracy_score(y_test, y_pred)

#If you want to test the result for just one input, just need to import like below:
#classifier.predict([[a, b, c, ....]])
#a, b, c, are inputs based on the excel file

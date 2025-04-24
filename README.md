# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: UDHAYA PRAKASH V
RegisterNumber:  212224240177
*/
```
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
## data.head()

![371445695-78518cdf-f83b-43ff-aac1-627bc90c493b](https://github.com/user-attachments/assets/bcaa5b53-716e-4196-a10a-025b445a26dc)

## data.info()

![371445907-f5411b66-b928-474f-8132-939ffe9fdef9](https://github.com/user-attachments/assets/e24955c2-459f-422d-b9a1-cd84af3bcae6)

## data.isnull().sum()

![371447292-d90f57fc-59eb-4f16-9dae-23e8401f3549](https://github.com/user-attachments/assets/13c6ec1c-71ec-4ba9-946c-58d38f9d7377)

## data.value.counts()

![371449083-8844545f-abcd-493e-80dd-3b6b50eae34b](https://github.com/user-attachments/assets/f4b5d6c5-eb41-4276-9bcf-687f235bfc8a)

## x.head()

![371446617-5ee6b2c5-e6c7-43e1-b872-bad8bd1415a4](https://github.com/user-attachments/assets/08258086-5ce5-4053-9d1d-4f14b92415ed)

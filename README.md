# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: John Wilfred Thomas J W
RegisterNumber: 24013517
*/
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv('Employee.csv')
print(data.head())
data.info()
print(data.isnull().sum())
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
print(data.head())
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','left','promotion_last_5years']]
print(x.head())
y=data[['left']]
print(y.head())
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])


## Output:
![decision tree classifier model](sam.png)
![image](https://github.com/user-attachments/assets/fdee9ef6-a44e-43f4-9027-e6ada402d182)
![image](https://github.com/user-attachments/assets/beb2c7d3-99c7-4b78-967b-995db12bf66d)
![image](https://github.com/user-attachments/assets/d727847b-4f92-4644-a3e8-2ad63b7bb884)
![image](https://github.com/user-attachments/assets/6e6c4c4f-ed9a-4d1a-ab31-5429c1a6d9eb)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

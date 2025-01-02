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
```

## Output:
![image](https://github.com/user-attachments/assets/aad35c2f-69af-4d96-819f-0877fd8e1910)
![image](https://github.com/user-attachments/assets/3dcf7515-92c8-4555-9e47-265aae16293a)
![image](https://github.com/user-attachments/assets/3d76f2de-69f2-4452-8a36-e47e5f9c2579)
![image](https://github.com/user-attachments/assets/7273a9f6-bc28-4dfe-8e21-797b3267ad9b)
![image](https://github.com/user-attachments/assets/c779f942-0e74-45a5-b58b-6812970c6c4a)
![image](https://github.com/user-attachments/assets/824df4a1-c053-4d0c-b251-315cf7103f15)
![image](https://github.com/user-attachments/assets/20b9422a-62af-4a07-93cb-fad314196ebb)
![image](https://github.com/user-attachments/assets/82a73e34-c170-4b8b-ad68-f745179bdd09)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

# Implementation of K-Means Clustering Algorithm
## Aim
To write a python program to implement K-Means Clustering Algorithm.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation

## Algorithm:

### Step1
<br>

### Step2
<br>

### Step3
<br>

### Step4
<br>

### Step5
<br>

## Program:
```
1.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
X1 = pd.read_csv("clustering.csv")
print (X1. head(2))
X2 = X1.loc[:, ['ApplicantIncome', 'LoanAmount' ]]
print(X2. head(2))
X = X2.values
sns.scatterplot(X[:,0], X[:, 1])
plt.xlabel('Income')
plt.ylabel('Loan')
plt.show( )
kmean=KMeans(n_clusters=4)
kmean. fit(X)
print('Cluster Centers: ',kmean.cluster_centers_)
print('Labels: ',kmean.labels_)
# predict the class for ApplicantIncome 9060 and Loanamount 120
predicted_class = kmean.predict([[9000, 120]])
print("The cluster group for Applicant Income 9060 and Loanamount 120 is",predicted_class)


2.
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
df = pd. read_csv("cars.csv")

X = df[[ 'Weight', 'Volume' ]]
y = df[ 'CO2']
regr = linear_model.LinearRegression( )
regr . fit(X, y)
# Cofficients and intercept of Model

print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)
#predict the CO2 emission of a car where the weight is 300kg, and the volume is 1300cm3:
predictedCO2 = regr . predict([[3300, 1300]])
print( 'Predicted CO2 for the corresponding weight and volume', predictedCO2)


```
## Output:

1.
![git](clustering.png)

2.
![git](cars.png)


### Insert your output

<br>

## Result
Thus the K-means clustering algorithm is implemented and predicted the cluster class using python program.

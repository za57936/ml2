import pandas as pd  #importing libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston  #loading dataset into boston object
boston = load_boston()

bos = pd.DataFrame(boston.data)      #loading data into bos dataframe
bos.columns = boston.feature_names   #renaming columns with it's feature names
bos['PRICE'] = boston.target     #adding the price column to the bos
print(bos.head())    #printing top 5 records
 
X = bos.iloc[:, :-1].values     #differentiating independent features & storing them in X

#differentiating dependent(output) features from dataframe & storing them in Y
y = bos.iloc[:, -1].values

#Splitting dataset into Training set & Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3)

#Fitting Simple Linear Regression to Training set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)   # Predicting Test set results
from sklearn.metrics import r2_score   #calculating accuracy of model using r2_score 
score=r2_score(y_test,y_pred)

#plotting expected value v/s predicted value
plt.scatter(y_test,y_pred,color='blue')
plt.title('expected value v/s predicted value')
plt.xlabel('expected value')
plt.ylabel('predicted value')
plt.show()
#  linear Regression :snowflake:
> In this project i used python to train MachineLearning Model.

` In this project i import following module `
<li>numpy
<li>pandas
<li>sklearn
 
```
  import numpy as np
  import pandas as pd
  import sklearn 
```
 
## WHAT I DO IN THIS PROJECT :question:
 
> First i import The Boston Housing Dataset. 

``` 
from sklearn.datasets import load_boston 
df = load_boston()
```
  
## WE LIST THE  KEY  :bookmark_tabs:
  
``` 
df.keys() 
```
> Following key of dataset.:point_down:
  
  <li>data
  <li>target
  <li>feature_name
  <li>DESCR
  <li>filename

## FREAMING DATASET :fireworks:
    
> Using the data and feature_names we create dataframe.
    
```    
boston = pd.DataFrame(df.data, columns=df.feature_names)
boston.head()
```

` Here head() print the top 5 record of the dataframe. `

|  NO  | CRIM    |	 ZN	 | INDUS | CHAS |	 NOX  |	  RM	|  AGE |   DIS	| RAD |   TAX   |	PTRATIO |	   B	 | LSTAT |
|------|---------|-------|-------|------|-------|-------|------|--------|-----|---------|---------|--------|-------|
|	 0   | 0.00632 |	18.0 | 2.31	 | 0.0	| 0.538 |	6.575 |	65.2 | 4.0900	| 1.0	|  296.0  |  15.3   | 396.90 | 4.98  |
|  1   | 0.02731 |	0.0	 | 7.07	 | 0.0	| 0.469 |	6.421 |	78.9 | 4.9671	| 2.0	|  242.0	|  17.8   | 396.90 | 9.14  |
|  2   | 0.02729 |	0.0	 | 7.07	 | 0.0	| 0.469 |	7.185 |	61.1 | 4.9671	| 2.0	|  242.0	|  17.8   | 392.83 | 4.03  |
|  3   | 0.03237 |	0.0	 | 2.18	 | 0.0	| 0.458 |	6.998 |	45.8 | 6.0622	| 3.0	|  222.0	|  18.7   | 394.63 | 2.94  |
|  4   | 0.06905 |	0.0	 | 2.18	 | 0.0	| 0.458 |	7.147 |	54.2 | 6.0622	| 3.0	|  222.0	|  18.7   | 396.90 | 5.33  |
  
  
 
## ADDING TARGET :small_red_triangle_down:
  
```
boston['MEDV'] = df.target
boston.head()
```
 
|  NO  | CRIM    |	 ZN	 | INDUS | CHAS |	 NOX  |	  RM	|  AGE |   DIS	| RAD |   TAX   |	PTRATIO |	   B	 | LSTAT | MEDV |
|------|---------|-------|-------|------|-------|-------|------|--------|-----|---------|---------|--------|-------|------|
|	 0   | 0.00632 |	18.0 | 2.31	 | 0.0	| 0.538 |	6.575 |	65.2 | 4.0900	| 1.0	|  296.0  |  15.3   | 396.90 | 4.98  | 24.0 |
|  1   | 0.02731 |	0.0	 | 7.07	 | 0.0	| 0.469 |	6.421 |	78.9 | 4.9671	| 2.0	|  242.0	|  17.8   | 396.90 | 9.14  | 21.6 |
|  2   | 0.02729 |	0.0	 | 7.07	 | 0.0	| 0.469 |	7.185 |	61.1 | 4.9671	| 2.0	|  242.0	|  17.8   | 392.83 | 4.03  | 34.7 |
|  3   | 0.03237 |	0.0	 | 2.18	 | 0.0	| 0.458 |	6.998 |	45.8 | 6.0622	| 3.0	|  222.0	|  18.7   | 394.63 | 2.94  | 33.4 |
|  4   | 0.06905 |	0.0	 | 2.18	 | 0.0	| 0.458 |	7.147 |	54.2 | 6.0622	| 3.0	|  222.0	|  18.7   | 396.90 | 5.33  | 36.2 |

## DEALING WITH NULL VALUE :collision: 
 
> Dataset may contain many rows and columns for which some values are missing. We can't leave those missing values as it is.
  
  1. Either drop the entire row or column.
  2. Fill the missing values with some appropriate value ex. mean of all the values for that column may do the job.

> Checking null values.
    
``` 
    boston.isnull()
    boston.isnull().sum()
```
    
> No null values.
    
### NOW WE GOOD TO GO FOR TRAIN THE MODEL :white_check_mark:
 
```
from sklearn.model_selection import train_test_split

X = boston.drop('MEDV', axis=1)
Y = boston['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state=5)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
  
```

> Here we first import train_test_split fromo sklearn.model_selection.

> Then we split the data in two part one is Target and other data.
    
> Then we divide whole set in two part one is train and second is test.
  
> We take size of test set as 0.15% of whole.So in this set 506 record are there from which we take 430 as train and 76 as test and we take it random.
  
 
### FITING LINER REGRESSION :bar_chart:
  
```
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    lin_model = LinearRegression()
    lin_model.fit(X_train, Y_train)
    
```

`We first import LinerRegression from sklearn.liner_model and mean_squared_error from sklearn.metrics .`
    
> Then fit it with train set.
    
## PREDICTION :mortar_board:
  
```
    y_train_predict = lin_model.predict(X_train)
    rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))

    print("The model performance for training set")
    print('RMSE is {}'.format(rmse))
    print("\n")

    # on testing set
    y_test_predict = lin_model.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

    print("The model performance for testing set")
    print('RMSE is {}'.format(rmse))
    
```

` FOR TRAIN SET `
    
> Here we predict y_train value for X_train. And then find error between y_train ( Predict value ) and Y_train ( Actual value ).
    
> After that we take squere root of that error. ---> ROOT MEAN SQUERE ERROR ( We Find ).
    
` SAME THING DO WITH TEST SET `
 
Hope you enjoy it :heart:.

PROFILE : [:snowman:](https://github.com/shiv2711)

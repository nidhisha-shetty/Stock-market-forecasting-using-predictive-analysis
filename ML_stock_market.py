import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('infy.csv')#reading data from infy.csv
df=df[['Open Price', 'High Price', 'Low Price','Close Price']]#considering 'open price','high price', 'low price', 'close price' columns from infy.csv file
df['label']=df['Close Price'].shift(-3) #creating a column 'label' and retrieving 3rd day from the current day to the current day 'label' column data; i.e predicting the 'close price' after 3 days in 'label' column
X=np.array(df.drop(['Close Price','label'],1)) #removing column names using '1' and also removing 'close price' and 'label' columns
X_lately=X[-3:] #creating data for which we will predict the 'close price' in the 'label' column
X=X[:-3] #creating 'X' train and test data from 'open price', 'high price', and 'low price'

y=np.array(df['label']) #creating array of data from 'label' column
y=y[:-3] #In the labels column, considering all the value till -3; #creating 'Y' train and test data from 'label' column

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8) #splitting X data into X_test and X_train in 80:20(0.8) ratio and splitting Y data into Y_test and Y_train in 80:20(0.8) ratio 

clf = LinearRegression() #using linear regression algorithm and feeding the algorithm with train and test data to predict the close price for next 3 days 
clf.fit(X_train, y_train)  #fit is a method within LR algo # assigning training set(80% of X_data i.e X_test and 80% of Y_data i.e Y_test)
print(clf)
confidence= clf.score(X_test, y_test)  #score is a method within LR algo # assigning testing set(20% of X_data i.e X_train and 20% of Y_data i.e Y_train )
print(confidence)
result=clf.predict(X_lately) #feeding the data that we created in step 10 to the algorithm to predict the values for the next 3 days based on clf.fit and confidence.
print(result)
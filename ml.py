#importing the libraries
import numpy as py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics 
#data collection and preprocessing
#loading the data to  a pandas dataframe
gold_data = pd.read_csv('C:\\Users\\Uday Verma\\Desktop\\Python AI Cllg\\gld_price_data.csv')
#print first 5 rows in the data frame
gold_data.head()
#print the last 5 rows in the data frame
gold_data.tail()
#number of rows and columns 
gold_data.shape
gold_data.info()
gold_data.isnull().sum()
#gettig the statistical measures of the data
gold_data.describe()
#correlation
#positive(one increase second increase) and negative(one increase then second decreases)
correlation = gold_data.corr()
#constructing heat map to understand correlation
plt.figure(figsize  = (8,8))
sns.heatmap(correlation, cbar = True, square  =True, fmt = '.2f', annot = True, annot_kws={'size':11}, cmap='Reds')
#cbar means bar (right side), fmt = decimal places , annot means text,square means shape, cmap is colour of map
#correlation values of GLD
print(correlation['GLD'])
#checking the distribution of GLD 
sns.displot(gold_data['GLD'],color ='green')
#splitting the features and target
x = gold_data.drop(['Date', 'GLD'],axis=1)
y= gold_data['GLD']
print(y)
#spltting into train and test data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
#model training random forest model
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(x_train, y_train)
#model evaluation , prediction on this data
test_data_prediction = regressor.predict(x_test)
print(test_data_prediction)
#r squared error
error_score = metrics.r2_score(y_test, test_data_prediction)
print("R sqaured is : ",error_score)
#compare the actual value and predicted value in a plot
y_test= list(y_test)
plt.plot(y_test, color ='blue', label = 'Actual value')
plt.plot(test_data_prediction, color = 'green', label = 'Predicted value')
plt.title('Actual price vs Predicted price')
plt.xlabel('Number of values')
plt.ylabel('GLD value')
plt.legend()
plt.show()

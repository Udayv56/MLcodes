#Data preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import math

df = pd.read_csv("homeprice.csv")
print(df)

median_bedroom = math.floor(df.bedroom.median())
print(median_bedroom)

df.bedroom = df.bedroom.fillna(median_bedroom)
print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['area','bedroom','age']],df.price)

# print(reg.coef_)

print(reg.predict([[3000,3,40]]))
print(reg.predict([[2000,4,5]]))

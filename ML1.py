import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("Book1.csv")
print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
plt.show()

area_to_predict = [[5000]]
predicted_price = reg.predict(area_to_predict)
print(predicted_price)

# print(reg.coef_)
# print(reg.intercept_)

d = pd.read_csv("area.csv")
print(d.head(5))

p = reg.predict(d)
d['prices'] = p

print(d)
d.to_csv("prediction.csv", index=False)

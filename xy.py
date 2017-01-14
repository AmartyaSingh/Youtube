import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import random
import numpy as np

#read data
df = pd.read_csv('challenge_dataset.txt')
xs = df[['X']]
ys = df[['Y']]

#train model on data
model = linear_model.LinearRegression()
model.fit(xs, ys)

#Taking Specific Example.
#test_x = 14.164
#predict_y = model.predict(test_x)
#actual_y = 15.505
#print ("For X_test = 14.164")
#print ("Y_predict = ", predict_y)
#print ("Y_Actual = ", 15.505)
#error = ((predict_y - actual_y) / actual_y)*100
#print ("Error(%) = ", abs(error))
#print ("Accuracy(%) for Example = ", 100-abs(error))

#Taking Random Example
expl = df.loc[random.sample(range(0, df.shape[0]), 1)]
rand_x = float(expl["X"])
rand_y = float(expl["Y"])

print ("Random X Example = %0.4f" %rand_x)
print ("Corresponding Y Example = %0.4f" %rand_y)
print ("Predicted Y = %0.4f" %model.predict(rand_x))

error_rand = abs((model.predict(rand_x) - rand_y) * 100 / rand_y)
print ("Percentage_Error = %0.4f" %error_rand)
print ("Percentage_Accuracy = %0.4f" %(100-error_rand))

#visualize results
plt.scatter(xs, ys)
plt.plot(xs, model.predict(xs))
plt.scatter(rand_x, model.predict(rand_x), color = 'r', s=50)
plt.scatter(rand_x, rand_y, color = 'g', s=50)
plt.show()

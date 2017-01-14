import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import random

#read data
dataframe = pd.read_csv('challenge_dataset.txt',header=None,names=["Brain","Body"])
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]


#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

# sample a random point and find actual and predicted error
sample = dataframe.loc[random.sample(range(0, dataframe.shape[0]), 1)]
rand_x = float(sample["Brain"])
rand_y = float(sample["Body"])
print("Sampled point Brain Size: %0.2f" %rand_x)
print("Sampled point Body Size: %0.2f" %rand_y)
print("Sampled point predicted Body Size: %0.2f" %body_reg.predict(rand_x))
print("Error: %.2f" %(body_reg.predict(rand_x) - rand_y))


#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.scatter(rand_x, rand_y, color='g')
plt.scatter(rand_x, body_reg.predict(rand_x), color='r')
plt.show()
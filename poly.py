# import packages to use for calculating data
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# assign values to train and tests
x_train = [[1], [2], [3], [4], [5], [6], [13],
           [22], [25]]  # days drove for deliveries
y_train = [[100], [99], [98], [88], [55], [
    99], [99], [82], [47]]  # distances covered

x_test = [[5], [3], [1], [2]]  # days drove for deliveries
y_test = [[90], [97], [98], [89]]  # distances covered

# plt.scatter(x_test, y_test, x_train, y_train)
# plt.show()

regression = LinearRegression()
regression.fit(x_train, y_train)
a = np.linspace(0, 25, 75)
b = regression.predict(a.reshape(a.shape[0], 1))
plt.plot(a, b)
plt.show()

# use the PolymonialFeature to assign the degree angle for the plotline
plotline = PolynomialFeatures(degree=4)

X_train_quadratic = plotline.fit_transform(x_train)
X_test_quadratic = plotline.transform(x_test)

regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = plotline.transform(a.reshape(a.shape[0], 1))

# plot the linespace, assign a colour and how it will look
# assign title for the plot
# assign label to x-axis
# assign label to y-axis
# assign how many numerical data both x- and y-axis must take and display
# display a grid for better viewing of the dataplots
# scatter the data on the plot
# show dataplot
plt.plot(a, regressor_quadratic.predict(xx_quadratic), c='b', linestyle='-')
plt.title('Distances covered by courier company')
plt.xlabel('Number of days')
plt.ylabel('Distance covered')
plt.axis([0, 30, 0, 150])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()
print(x_train)
print(x_test)
print(X_train_quadratic)
print(X_test_quadratic)

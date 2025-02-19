import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class RecursiveLeastSquares:
    def __init__(self, numberFeatures):
        """
        For the bias term, add one to the number of features.
        """

        self.theta = np.zeros((numberFeatures + 1, 1))
        self.P = np.eye(numberFeatures + 1)

    def step(self, x, y):
        """
        Update the estimate.

        Arguments:
        x: A new entry array of features.
        y: The corresponding label.

        Returns:
        The updated estimate (theta).
        """

        self.P = self.P - np.dot(np.dot(self.P, x), np.dot(x.T, self.P)) / (1 + np.dot(np.dot(x.T, self.P), x))
        K = np.dot(self.P, x)
        self.theta = self.theta + K * (y - np.dot(x.T, self.theta))

        return self.theta

f = pd.read_csv('2Dim.csv')
X = np.array(f['X2 house age'])
Y = np.array(f['Y house price of unit area'])

model = RecursiveLeastSquares(1)
theta = np.zeros((2, 1))

plt.scatter(X, Y)

for k in range(X.shape[0]):
    x = np.array([[1], [X[k]]])
    y = Y[k]
    theta = model.step(x, y)

    # Update the Plot with the new regression
    Y_hat = [theta[0] + theta[1] * x for x in X]

    if k == X.shape[0] - 1:
        plt.plot(X, Y_hat, color='blue')
    else:
        plt.plot(X, Y_hat, color='red')

    plt.pause(0.1)

plt.show()

# Compare with the sklit result to check the correctness of the implementation

X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

model = LinearRegression()
model.fit(X, Y)

# Plot the result
plt.scatter(X, Y)
plt.plot(X, model.predict(X), color='blue')
plt.show()

print(theta)

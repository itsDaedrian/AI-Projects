import numpy as np
import matplotlib.pyplot as plt

def linear_regression(df, learning_rate, max_iterations):

    m = df.shape[0]

    X = np.c_[np.ones((m, 1)), df.iloc[:, 0].values]

    y = df.iloc[:, 1].values.reshape(-1, 1)

    theta = np.zeros((2, 1))

    cost_history = []
    # Iterate through the specified number of iterations
    for i in range(max_iterations):

        h = X.dot(theta)

        cost = 1/(2*m) * np.sum((h-y)**2)

        gradient = 1/m * X.T.dot(h-y)
        # Update the theta values
        theta = theta - learning_rate * gradient

        cost_history.append(cost)
        # Print the cost every 5 iterations
        if i % 5 == 0:
            print("Iteration", i, "- Cost:", cost)
            # Plot the data and regression line every other 5 iterations
            if i % 10 == 0:
                plt.scatter(df.iloc[:, 0], df.iloc[:, 1], color='blue')
                plt.plot(df.iloc[:, 0], X.dot(theta), color='red')
                plt.title("Iteration " + str(i))
                plt.show()
        # Stop if the cost does not change much between iterations
        if i > 0 and abs(cost_history[-1] - cost_history[-2]) < 1e-7:
            print("Converged in", i, "iterations.")
            break
    # Plot the final data and regression line
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], color='blue')
    plt.plot(df.iloc[:, 0], X.dot(theta), color='red')
    plt.title("Final Regression Line")
    plt.show()
    return theta


import pandas as pd

# Get input
m = int(input("Enter the number of training examples: "))
data = np.zeros((m, 2))
for i in range(m):
    data[i, :] = input("Enter x and y values separated by a space: ").split()
df = pd.DataFrame(data, columns=['X', 'y'])

# Get learning rate and maximum number of iterations
learning_rate = float(input("Enter the learning rate: "))
max_iterations = int(input("Enter the maximum number of iterations: "))


theta = linear_regression(df, learning_rate=learning_rate, max_iterations=max_iterations)


print("Learned parameters:")
print("Intercept:", theta[0])
print("Slope:", theta[1])

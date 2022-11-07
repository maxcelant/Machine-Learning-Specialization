import numpy as np

def cost_function(X, y, w, b):
  total_cost = 0
  for i in range(m):                                           # simplified cost function
    z = np.dot(w, X[i]) + b                                    # calculate prediction z(i)
    g = 1 / (1 + np.exp(-z))                                   # use sigmoid func to get g(i)
    y_hat = g                                                  # prediction equal to g(z(i))
    loss = -y[i] * np.log(y_hat) - ((1 - y[i]) * np.log(1 - y_hat)) # loss func
    total_cost += loss                                         # add to total cost
  total_cost =/ m                                              # divide by number of examples
  return total_cost

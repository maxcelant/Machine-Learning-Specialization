- Using the slope of a line to help find a good prediction
- **Regression** model predicts **numbers**
- **Classification** model predicts **categories**
- Using the training set, the supervised learning algorithm will produce a function `f` also called the **model**
	- The function takes in input `x` and returns `y^` or **y-hat**, which is the prediction or **(estimated y)**
- We can represent a simple linear model with `f(x) = wx + b`, where `x` is the input `f(x)` is y-hat

![[Pasted image 20221102113057.png]]

#### Terminology
- **Training Set**: data set used to train the model
- `x` is called the **input** variable or **feature**
- `y` is called the **output** variable  or **target** variable
	- if `x = 2104 (size in feet)` then y will be `400 (price in $1000's)`
- `(x(i),y(i))` is a single training example, which is the `i-th` training example
	- `(2104, 400)` is the single training set when `i = 1`
	- `i` refers to a specific row in the training data set
- **Univariate Linear Regression** means linear regression with one input variable.

![[Pasted image 20221102113833.png]]

#### Lab

###### Notation
- `x_train`: training example feature values
- `y_train`: training example target values
- `m`: Number of training examples
- `w`: parameter - weight
- `b`: parameter - bias

```py
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 2.0])
y_traub = np.array([300.0, 500.0])

m = x_train.shape[0]

# To get i-th training example
i = 0
x_i = x_train[i]
y_i = y_train[i]

# Plotting the data
plt.scatter(x_train, y_train, market='x', c='r')
plt.title("Housing Prices")
plt.ylabel("Price (in 1000s of dollars)")
plt.xlabel("Size (1000 sqft)")
plt.show()

# Model function
w = 100 # (these do not give a good line that fits this model)
b = 100 # (these do not give a good line that fits this model)

def compute_model_output(x, w, b):
	"""
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
	m = x.shape[0]
	f_wb = np.zeroes(m)
	for i in range(m):
		f_wb[0] = w * x[i] + b
	return f_wb

tmp_f_wb = compute_model_output(x_train, w, b)

# Plot our prediction model
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

# It turns out w = 200, b = 100 is the are the best values for this model
w, b = 200, 100

# Predict the cost of a house that has 1,200 square feet
x_i = 1.2
y_hat = w * x_i + b
# Prediction
print(f"${y_hat:0.f} thousand dollars")
```
- We can't use a straight line in linear regression
- Uses the **Sigmoid Function** or also called the **Logistic Function**
	- outputs values between 0 and 1

$$
f(\vec{x}) = g(z) = \frac{1}{1+e^{-z}}
$$

![[Pasted image 20221106094121.png|center]]

- The Logistic Function is calculated like so:
![[Pasted image 20221106094224.png|center]]
- Basically we insert our slope function into the sigmoid function
$$
f(\vec{x}) = g(\vec{w} • \vec{x} + b) = \frac{1}{1+e^{-(\vec{w} • \vec{x} + b)}}
$$

- Interpret as the "probability" that class is 1
- if `f(x) = 0.7`, this means that the model things its a 70% chance that `y = 1`
- **"What is the probability that y is 1, given input x and parameters w,b"** ![[Pasted image 20221106094749.png]] 

- In NumPy, we can compute it like so:
```py
def sigmoid(z: ndarray):
	g = 1/(1 + np.exp(-z))
	return g
```
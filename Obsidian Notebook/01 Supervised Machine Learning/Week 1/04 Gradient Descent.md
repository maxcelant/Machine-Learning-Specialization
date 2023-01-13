- Can be used to minimize ANY function, recall that minimizing means getting the most accurate output for your model
- **Outline**
	- Start with some `w` and `b` (usually set w=0, b=0)
	- Keep changing `w` and `b` to reduce `J(w,b)`
	- Do this until we settle at or near a minimum (note that there could be more than 1 minimum)
		- ![[Pasted image 20221103113651.png]]
- **Local Minima** is the term to describe multiple minimums in a single map
	- ![[Pasted image 20221103113826.png]]
- **"Batch" Gradient Descent** means that each step of gradient descent uses all the training examples

---

#### Gradient Descent Algorithm

- You perform this algorithm with both `w` and `b` **simultaneously**

$$
w = w - \alpha * \frac{d}{dw} J(w,b)
$$

$$
b = b - \alpha * \frac{d}{db} J(w,b)
$$

- We do this until both `w` and `b` dont move very much
- ![[Pasted image 20221103114832.png]] is the **Learning Rate**, it controls how large the step you take down is
	- The larger alpha is, the more agrressive the Gradient Descent procedure is
- ![[Pasted image 20221103114850.png]] tells you which direction you are taking your step in combination with the learning rate

---

#### Intuition
- Taking the derivative gives us a tangent line at the angle of the point
- If the slope is negative, then this makes us move to the right
- If its positive, we move to the left
![[Pasted image 20221103120259.png]]

---

#### Learning Rate
- If the learning rate (alpha) is too small, then gradient descent may be too **slow**
- If the learning rate is too large, gradient descent may **overshoot**, never reach minimum, fail to converge
- As you get closer to a local minimum, gradient descent will automatically take smaller steps
	- Derivative becomes smaller
	- Update steps become smaller
	- ![[Pasted image 20221103121210.png]]

---

#### Gradient Descent for Linear Regression
- ![[Pasted image 20221103121639.png]]

---

#### Lab

```py
# for all the inputs, compute the total cost/error
# note that we don't actually use this formula in the code below, but it is important
# that we understand that this is how we calculate how where on the parabola
# a certain w and b combination lies on the J(w,b) cost function
def compute_cost(x, y, w, b):
    m = len(x)
    cost = 0
    for i in range(m):
        y_hat = w * x[i] + b
        cost = cost + (y_hat - y[i])**2
    total_cost = 1 / (2 * m) * cost
    return total_cost


# this actually performs the derivative of both b and w (simultaneously)
def compute_gradient(x, y, w, b):
    m = len(x)
    dj_dw = 0
    dj_db = 0
    # compute the sum from 0 to m
    # then divide that total sum by m
    # we will then multiply these values by the learning rate in the gradient descent formula
    for i in range(m):
        y_hat = w * x[i] + b
        dj_dw_i = (y_hat - y[i]) * x[i]
        dj_db_i = (y_hat - y[i])
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

  
# performs the gradient descent algorithm using a specified learning rate
# and number of epochs
def gradient_descent(x, y, w_in, b_in, alpha, epochs, compute_gradient):
    b = b_in
    w = w_in
    for i in range(epochs):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        # recall that alpha is the learning rate
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
    return w, b
  
  
def main():
    x = [1.0, 2.0] # features
    y = [300.0, 500.0] # targets

    w = 50
    b = 50

    alpha = 1.0e-2 # learning rate
    epochs = 10000 # number of iterations
    w_final, b_final = gradient_descent(x, y, w, b, alpha, epochs, compute_gradient)
    print(f'w: {w_final}, b: {b_final}')
    # best approximation = w_final * x + b_final


if __name__ == '__main__':
    main()
```
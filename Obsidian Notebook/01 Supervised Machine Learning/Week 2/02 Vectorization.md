- If we didn't use vectorization, our code would look like this and be very slow
```py
w = np.array([1.0, 2.5, -3.3])
b = 4
x = np.array([10, 20, 30])

# without vectorization...
f = w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + b 
```
- With vectorization, however, its much simpler and faster
- Recall that **dot-product** literally does what the code above does, but faster
	- multiply the pair and add them all together
```py
f = np.dot(w, x) + b
```

- Lets say we had a model with 16 parameters (`w`)
- Without vectorization, this would take forever, but utilizing NumPy, this is made easy
```py
import numpy as np

w = np.array([0.5, 1.3, ..., 3.4])
b = np.array([0.3, 0.3, ..., 2.1])

# without vectorization...
learning_rate = 0.1
for j in range(0, 16):
	w[j] = w[j] - learning_rate * d[j]

# with vectorization! So simple!
w = w - learning_rate * d
```

---

#### Lab
```py
import numpy as np

a = np.zeroes(4)                # a = [0, 0, 0, 0]
a = np.random.random_samples(4) # a = [0.36826642 0.47204354 0.2668734  0.81426875]
a = np.arange(4)                # 1 = [0, 1, 2, 3]
a.shape                         # (4,)

# Slicing -> [start:end:step]
c = a[2:7:1]                    # [2,3,4,5,6]
c = a[2:7:2]                    # [2,5,6]

# Operations
b = -a                          # [-1, -2, -3, -4]
b = np.sum(a)                   # 10
b = np.mean(a)                  # 2.5
b = a**2                        # [1, 4, 9, 16]

# Element-wise operation
a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
c = a + b                       # [0 0 6 8]

# Scalar operations
b = 5 * a                       # [5, 10, 15, 20]

# dot product

a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c = np.dot(a, b)

```
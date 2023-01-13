- What if we wanted to implement a layer of a neural network ourself? 
- `w2_1` means 2nd layer, 1st neuron
![[Pasted image 20221111093955.png|center]]
```python
x = np.array([200, 17]) # input vector

# 1st neuron in 1st layer
w1_1 = np.array([1, 2]) # weights
b1_1 = np.array([-1])
z1_1 = np.dot(w1_1, x) + b1_1
a1_1 = sigmoid(z1_1)

# 2nd neuron in 1st layer
w1_2 = np.array([-3, 4]) 
b1_2 = np.array([1])
z1_2 = np.dot(w1_2, x) + b1_2
a1_2 = sigmoid(z1_2)

# 3rd neuron in 1st layer
w1_3 = np.array([5, -6])
b1_3 = np.array([2])
z1_3 = np.dot(w1_3, x) + b1_3
a1_3 = sigmoid(z1_3)

# output from 1st layer!
a1 = np.array([a1_1, a1_2, a1_3])

# 1st neuron in 2nd layer
w2_1 = np.array([-7, 8, 9])
b2_1 = np.array([3])
z2_1 = np.dot(w2_1, a1) + b2_1 
a2 = 
```
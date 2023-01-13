![[Pasted image 20221107104147.png|center]]

- We basically combine both formulas from the previous page [[04 Cost Function for Logistic Regression]]
- **Recall** that the cost function is used for linear regression!
- The equation above is important because it allows us to combine both equations **into one**
- When y = 1, the `Loss(y_hat, y)` is the first term
![[Pasted image 20221107104322.png]]
- When y = 0, the `Loss(y_hat, y)` is the second term 
![[Pasted image 20221107104305.png]]

---

#### Simplified Cost Function

![[Pasted image 20221107104553.png]]

#### Simplified Cost Function In Code
```py
import numpy as np

total_cost = 0
for i in range(m):                                           # simplified cost function
	z = np.dot(w, X[i]) + b                                  # calculate prediction z(i)
	g = 1 / (1 + np.exp(-z))                                 # use sigmoid func to get g(i)
	y_hat = g                                                # prediction equal to g(z(i))
	loss = -y[i] * np.log(y_hat) - ((1 - y[i]) * np.log(1 - y_hat)) # loss func
	total_cost += loss                                       # add to total cost
total_cost =/ m                                              # divide by number of examples
return total_cost
```
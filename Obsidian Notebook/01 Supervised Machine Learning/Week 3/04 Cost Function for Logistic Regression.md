- Squared error cost function does not work well for logistic regression
- Using the logistic loss function, we can use gradient descent like with our other model
- `Loss(y_hat, y)`

##### Logistic Loss Function
"**The further prediction `f_w_b(x)` or `y_hat` is from the target `y`, the higher the loss**"
![[Pasted image 20221106120530.png|center]]

The overall cost function looks like so:

![[Pasted image 20221106121816.png]]

---

##### For y = 1...
- we only care about this part of the graph [`-log(f(x))`]
 ![[Pasted image 20221106120613.png]]
- If `y_hat` or `f_w_b(x)` predicts close to 1, then the loss is very small (close to 0)
- If the loss is closer to 0 even though true y = 1, then the loss is very high! (infinity)
![[Pasted image 20221106120911.png]]

---
##### For y = 0...
- the graph looks like this [`-log(1 - f(x))`]
![[Pasted image 20221106121121.png]]
- if `y_hat` or `f_w_b(x)` predicts close to 0, then the loss will also be close to zero (small loss)
- If the loss is closer to 1 even though true y = 0, then the loss is very high! (infinity)
![[Pasted image 20221106121528.png]]
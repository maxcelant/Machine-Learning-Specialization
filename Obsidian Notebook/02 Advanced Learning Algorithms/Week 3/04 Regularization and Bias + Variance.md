- Recall what Regularization does:  [[09 Addressing Overfitting]]
- If your regularization variable **`λ`** is too high, you will most likely underfit the data.
- If your regularization variable **`λ`** is too low, then it won't really change the data at all.
- We can use the cross-validation set to find a good value of lambda, **`λ`**.
- Let's see this as a graph, where the X-axis is the **`λ`** and the Y-axis is the cost.
![[Pasted image 20221126171534.png|center]]
- If **`λ`** is too large or too small, then it doesn't do well on the CV set 
- However, notice how if **`λ`** is low, the training set will do well because it is overfitting the data!


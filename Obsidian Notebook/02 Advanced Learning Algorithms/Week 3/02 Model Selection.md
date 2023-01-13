- The training error `J(w,b)` will likely be much lower than the test error `J(w,b)`/ the actual generalization error
- You can test your model with different order polynomials to see which one gives you the highest accuracy
![[Pasted image 20221119225432.png|center]]

- Instead of separating our dataset into 2 groups, we can separate it into three sets
	- 60% training set
	- 20% cross validation set (also called **dev set**)
	- 20% test set
- We use the Cross Validation set to find the best fitting order of polynomial for our set, which we will then use to test our test set
![[Pasted image 20221119230219.png|center]]

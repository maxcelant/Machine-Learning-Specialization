- To find out if your model has high or low variance, pay attention to the training set and validation set
- If it doesn't do well on the training set, then theres a good chance it the model has **high bias (underfit)**
- If it does REALLY well on training set but not so good on the validation set, then the model is most likely **high variance (overfitting)** the data
	- Ex: training set: 93%, validation set: 86%; its a high variance problem
![[Pasted image 20221126165628.png|center]]

- You are looking for the sweet spot where the validation set or `J_cv` is the lowest

![[Pasted image 20221126165847.png|center]]

- If a categorical feature can take on `k` values, create `k` binary features (0 or 1 valued)
	 **Example:** if a cat can have pointy, floppy or oval ears, then turn it into three different features. Pointy: True/False, Floppy: True/False, Oval: True/False 
![[Pasted image 20221127145310.png|center]]
- You should encode each feature as 0 or 1, 1 if you have it, and 0 if you don't 

### Continuous Valued Features
- What about things like weight? It can be any positive value
- The ideal way would be to split the weight at the highest information gain
![[Pasted image 20221127145724.png|center]]
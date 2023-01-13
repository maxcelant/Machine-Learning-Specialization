- What if we had more than 1 input feature?
- We use subscripts to denote each feature, `x˅j` to denote the `j-th` feature
- `n = total number of features`
- `x^i = features of i-th training example`
	- Sometimes an arrow will be placed on the `x` to denote its a whole row or vector 
	- Think of this as a row in a table, all the features that correspond to the same entity
	- ![[Pasted image 20221103160604.png]]
- if it has a superscript `i` and a subscript `j` then it denotes the `i-th` row and the `j-th` column cell
- `(i, j) -> (row, col)`
- This is how we might denote it now with more features
	- ![[Pasted image 20221103161218.png]]
	- **Example:**
	- ![[Pasted image 20221103161241.png]]
- We can denote ![[Pasted image 20221103161526.png]] ` = [w1, w2, w3, ... wn]`, the arrow simply means its a vector (which is just a list of numbers), if we add `b` to this, these are the **parameters of the model**
- Recall that `x` with the arrow is also a vector of all the input features
- So we can rewrite the equation as:

![[Pasted image 20221103161823.png|center]]

- This is called **Multiple Linear Regression**

- Input layer is not usually counted as a layer

![[Pasted image 20221107232325.png|center]]

- In the example above, layer 3 takes the `a^[2]` as input and 
	- each neuron performs the `a^[3] = g(w • a^[2] + b )`
	- Notice how we use the **sigmoid activation function** here
- Notice how each neuron calculates just one value that will become the new vector, `a^[3]`
	- ![[Pasted image 20221107232625.png]]
- This can be denoted more simply like so

![[Pasted image 20221107233148.png|center]]
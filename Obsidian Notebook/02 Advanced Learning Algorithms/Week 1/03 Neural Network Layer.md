- Each neuron has its own logistic regression algorithm with unique `w` and `b` values
- The input to each neuron comes as a vector of values from the previous layer
- The outputs of the layer are then passed as a vector to the next layer, usually denoted with 
![[Pasted image 20221107231616.png|center]]

![[Pasted image 20221107231318.png|center]]

- **Note:** Layers are denoted using square brackets, `[n]`
	- They are usually place as a superscript to the value
- **Note:** Neuron number is denoted in the subscript location, with the top neuron in the layer being '1'
![[Pasted image 20221107232923.png|center]]
- Layer numbering starts with 0 being the input layer and increasing from there
- The inputs to the final layer would be `a^[1] = [0.3, 0.7, 0.2]`
- The computed output is `a^[2] = 0.84`
	- if `a^[2] >= 0.5`, yes, else no

![[Pasted image 20221107231803.png|center]]
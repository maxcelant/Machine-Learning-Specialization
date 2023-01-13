#### Dense Layer
- The one we've been using
- Each neuron output is a function of all the activations outputs of the previous layer

#### Convolutional Layer
- Each neuron only looks at a region of the 'image' or object
- Each neuron only looks at part of the previous layer's inputs
![[Pasted image 20221118095911.png|center]]
- **Why?**
	- Faster computing
	- Needs less data
	- Less prone to overfitting
- You could have the areas that each neuron is looking at overlap with each other.
- You can have the next layer also a convolutional layer (usually called a Convolutional NN), where there are less neurons, but each nueron only looks at a subpart of the previous layer
![[Pasted image 20221118100432.png|center]]


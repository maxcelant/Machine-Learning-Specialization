- **Cost Function (J)** tells us how well the model is doing by comparing the prediction outputs to the true values
- Essentially, we want to **minimize** J with respect to w and b
- Does this by comparing the prediction `y-hat (predicted output)` to `y (actual output)`
	- We sum all the errors of all the points given
	- `predicted output (y-hat)- actual output (y)` and then square it
	- `m = number of training examples`, so we do this for all training examples
	- `1/2m gives us the neat average square error`
	- `J = cost function or squared error cost function`
	- ![[Pasted image 20221102153439.png]]
- `f(x) = wx + b`, `w` and `b` are parameters/variables you can adjust during the training in order to improve the model.
	- `w` and `b` may also be referred to as **coefficients** or **weights**
	- `w` is the slope
- 

![[Pasted image 20221102152540.png|center]]

#### Calculating Cost Function
- Here we calculate the cost if `w = 0.5`, as you can see, its not very accurate
- `J(0.5) = 1/2m[(0.5 - 1)^2 + (1-2)^2 + (1.5 + 3)^2]`
- `J` ends up being equal to `0.58`
	- ![[Pasted image 20221102161154.png]]
- What we end up seeing is that we can plot all the points of `w` to `J(w)` which creates a parabola! We want to choose the point where the cost `J` is at or near a minimum! This will give us model that fits the data the best.
	- ![[Pasted image 20221102161615.png]]
- Since a normal cost function will have both `w` and `b`, it will look more like this
	- ![[Pasted image 20221102163911.png]]
- We can actually turn this 3-D plot into a 2-D plot by using contour lines like you would for geography. The best choice for `w` and `b` can be found by going to the minimum of the J graph!
	- ![[Pasted image 20221102165028.png]]
- Here is an example of an ideal prediction line
	- ![[Pasted image 20221102165518.png]]
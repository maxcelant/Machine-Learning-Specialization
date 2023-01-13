- We can use a learning curve to see how many iterations or **epochs** it takes for the machine to learn

![[Pasted image 20221105111955.png|center]]

- You can see the cost of J at each iteration
- `J(w,b)` should always decrease with each iteration, if it doesn't that means you need to change your alpha (learning rate) value 
- **Automatic Convergence** is using a value `ε = 0.001`, if `J(w,b)` decreases by <= `ε`, in one iteration, then we can declare convergence
- We can use alternative activation functions that aren't `sigmoid`
- If we need the value to be greater than 0 to 1
- This activation is called `ReLU` or Rectified Linear Unit
- `g(z) = max(z, 0)`, this means that the least it could be is 0
- In this example, the derived "awareness" feature is not binary but has a continuous range of values. The sigmoid is best for on/off or binary situations. The ReLU provides a continuous linear relationship. Additionally it has an 'off' range where the output is zero.

![[Pasted image 20221112102516.png|center]]

- The **linear activation function** is also known as "no activation function"
![[Pasted image 20221112102752.png|center]]
- If it's not doing well on the training set then you need to add more hidden layers
- If it doesn't do well on the validation set, then we need to get more data
![[Pasted image 20221126223358.png|center]]
- A larger neural network will usually do better than a smaller one, assuming regularization is chosen appropriately
```python
layer_1 = Dense(25, activation='relu', kernel_regularizer=L2(0.01))
```
- `kernel_regularizer=L2(0.01)` is the lambda regularization value
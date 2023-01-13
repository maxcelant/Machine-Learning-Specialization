#### Output Layer
- If you are doing **Binary Classification**, use sigmoid activation function
- If you are doing **Regression** and y can be *positive or negative*, use linear activation function
- If you are doing **Regression** and y can only be *positive or 0*, use ReLU activation function

#### Hidden Layer
- The most common choice is to use **ReLU** activation for hidden layers
```python
Dense(units=25, activation='relu')
```
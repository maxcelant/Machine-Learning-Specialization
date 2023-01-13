- We are going into more detail on how we train neural networks!

```python
# 1. Specify the model
model = Sequential([
        Dense(3, activation='sigmoid', name = 'layer1'),
        Dense(1, activation='sigmoid', name = 'layer2')
])

from tensorflow.keras.losses import BinaryCrossentropy

# 2. Compiles the model
model.compile(loss=BinaryCrossentropy)

# 3. Trains the model
model.fit(X, Y, epochs=100)
```


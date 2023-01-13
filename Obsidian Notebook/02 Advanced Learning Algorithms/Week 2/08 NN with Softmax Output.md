- If we have 10 possible outputs, then the output layer will have 10 neurons and use the `softmax` activation function.
![[Pasted image 20221112151237.png|center]]
- Each neuron in the final layer will be the estimate that `y = n`, `n` being 1 of the 10 possible classes

#### Implementation in the code...
```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
        Dense(25, activation='relu'),
        Dense(15, activation='relu')
        Dense(10, activation='softmax')
])
```

- Since this is a multiclass problem and not binary classification, we need to pick a different loss function!
- `Sparse` meaning it can only be one of the categories, `Categorical` meaning it will be in categories still
- **NOTE** there is a better way to write this code in tensorflow!!
```python
from tensorflow.keras.losses import SparseCategoricalCrossentropy

model.compile(loss=SparseCategoricalCrossentropy())
```

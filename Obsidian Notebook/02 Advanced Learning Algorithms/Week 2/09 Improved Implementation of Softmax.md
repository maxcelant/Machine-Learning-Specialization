- The computer can have numerical round off error!
![[Pasted image 20221112152230.png|center]]
- Instead of calculating `a` separately then plugging it in, we can directly compute `1 / 1 + e^(-z)` with in the loss function to get better accuracy!
- `logit = z`
![[Pasted image 20221112152521.png]]

---
#### With Logistic Regression...
```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
        Dense(units=25, activation='relu'),
        Dense(units=15, activation='relu')
        Dense(units=1, activation='linear')
])

model.compile(loss=BinaryCrossEntropy(from_logits=True))

model.fit(X, Y, epochs=100)

logits = model(X)

f_x = tf.nn.sigmoid(logits)
```

---
#### With Softmax...
- This is more accurate!
- Note that we use `linear` as the activation function here
```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
        Dense(units=25, activation='relu'),
        Dense(units=15, activation='relu')
        Dense(units=10, activation='linear')
])

model.compile(loss=SparseCategoricalCrossEntropython(from_logits=True))

model.fit(X, Y, epochs=100)

logits = model(X)

f_x = tf.nn.softmax(logits)
```


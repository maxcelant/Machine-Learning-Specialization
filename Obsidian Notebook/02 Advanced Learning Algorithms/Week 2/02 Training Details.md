- Just like logistic regression, neural networks have the same 3 steps even though they may look different

![[Pasted image 20221112101102.png|center]]

#### Loss and Cost Function
- The function we've been using in logistic regression is the same one here called **Binary Crossentropy**, It's called "binary" because it is a binary classification problem
![[Pasted image 20221112101353.png|center]]
- You can read more [here](obsidian://open?vault=Machine%20Learning&file=01%20Supervised%20Machine%20Learning%2FWeek%203%2F05%20Simplified%20Cost%20Function)

```python
from tensorflow.keras.losses import BinaryCrossentropy

model.compile(loss=BinaryCrossentropy())
```

- You can also perform linear regression (to guess a number), just use mean squared error function
```python
from tensorflow.keras.losses import MeanSquaredError

model.compile(loss=MeanSquaredError())
```
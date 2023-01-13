- "Adam" (Adaptive Moment Estimation) algorithm takes bigger steps towards the minima in Gradient Descent if it notices that alpha is too small.
- This algorithm can also make it smaller if your learning rate (alpha) is too high
- It uses different learning rates for each parameter of the model

#### Intuition
- If `w` keeps moving in the same direction, increase `a[j]`
- if `w` keeps oscillating, then decrease `a[j]`

#### In Code...
```python
model.compile(
	optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)
```
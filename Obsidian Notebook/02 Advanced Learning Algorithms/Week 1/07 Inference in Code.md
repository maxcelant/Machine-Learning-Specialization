![[Pasted image 20221108113727.png|center]]

- This image can be written in TensorFlow like so:
```py
x = np.array([[200.0, 17.0]])
layer_1 = Dense(units=3, activation='sigmoid')
a1 = layer_1(x)

layer_2 = Dense(units=1, activation='sigmoid')
a2 = layer_2(a1)

yhat = 1 if a2 >= 0.5 else 0
```
- `layer_1` relates to the first layer in the image, with `units` being the number of neurons
- `yhat` is the output prediction
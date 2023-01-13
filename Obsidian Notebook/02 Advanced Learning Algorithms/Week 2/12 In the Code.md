```py
test_example = X[1000] # this example is a 2

prediction_p = model.predict(test_example.reshape(1,400)) # turn it into a single column vec

# argmax returns an integer representing the highest probability
yhat = np.argmax(prediction_p)
```
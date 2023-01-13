#### Dataset
```py
import numpy as np

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])
```

#### Fit the Model
- Think of `.fit(X, y)` as creating the curve for the model
```py
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X, y)
```

#### Make Predictions
```py
y_pred = lr_model.predict(X)

print("Prediction on training set:", y_pred)
```
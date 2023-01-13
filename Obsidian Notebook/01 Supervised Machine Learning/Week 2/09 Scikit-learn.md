**Tool importing**
```py
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
```

**Import data**
```py
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']
```

**Use scikit to normalize the training data**
```py
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
```

**Create and fit regression model**
```py
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
```

**View Parameters**
```py
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
```

**Make Prediction**
```py
# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm

print(f"Prediction on training set:\n{y_pred[:4]}" ) # Prediction: [295.21 485.92 389.62 492.09]
print(f"Target values \n{y_train[:4]}") # Actual: [300.  509.8 394.  540. ]
```
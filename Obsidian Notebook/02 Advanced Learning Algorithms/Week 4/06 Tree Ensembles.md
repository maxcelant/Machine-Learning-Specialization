- Using multiple decision trees is called a **Tree Ensemble**.
- If you use multiple trees, it can give you more accurate predicitions.

### Sampling with Replacement
- Picking something from a group, then putting it back in.
- This allows you to create a new different training set based on your original training set.

### Random Forest Algorithm
- Given a training set of size `m`
- For `b = 1` to `B (total trees we want to create)`:
	- Use a sampling with replacement to create a new training set of size `m`
	- Train a decision tree on the new dataset
- **Note:** Changing the dataset will cause the nodes to show up differently because the information gain may be better for one split over another.
- At each node, when choosing a feature to use to split, if `n` features are available, pick a random subset of `k < n` features and allow the algorithm to only choose from that subset of features.
	- `k` is typically `sqrt(n)`

### XGBoost
- Works similar to random forest algorithm, except...
- Instead of picking from all examples with equal (1/m) probability, make it more likely to pick **misclassified examples** from previously trained trees
- **XGBoost (eXtreme Gradient Boosting)**
	- Open source implementation of boosted trees
	- Fast efficient implementation

### Classification
```python
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

### Regression
```python
from xgboost import XGBRegressor

model = XGBRegressor()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

### When Should You Use a Decision Trees?
- **Decision Trees / Tree Ensembles**
	- Work well on tabular (structured) data
		- Like spreadsheets
	- Not good for images, audio, text
	- Fast
	- Human interpretable
- **Neural Networks**
	- Works on structured and unstructured data
	- Slower
	- Works with transfer learning
### Reducing the Number of Features
- Take data with a lot of features and reduce it to visualize the data
![[Pasted image 20230107113618.png|center]]
- Take n features and compress them down to 2 or 3 features usually denoted `z_1`, `z_2`, etc

### PCA Algorithm
- Preprocess features: 
	- Normalize to have zero mean
	- [[04 Feature Scaling]]
- Choose an axis, and capture just one value from those coordinates
![[Pasted image 20230107114355.png|center]]
- What you notice is that the *variance* is quite large between the points on the z-axis
![[Pasted image 20230107114540.png|center]]
- However, in this axis, the points are very squished together, the *variance* is much less.
![[Pasted image 20230107114624.png|center]]
- This would be the best choice!
- This line is called the *principal component* with max variance

- How do we project a coordinate `x` vector onto this new z-axis?
	- We can think of this new `z` axis as a vector with `[0.71 0.71]`
	- `x`'s coordinates are `[2 3]`
	- To get the project, we need to dot product `x` and `z`

![[Pasted image 20230107115049.png|center]]


### PCA in Code
- Optional: preprocess with feature scaling
- "fit" the data to obtain 2 (or 3) new axes (principal components)
	- This part includes mean normalization
- Examine how much variance (info) is explained by each principal component
- Transform (project) the data onto the new axes

```python
X = np.array([[1,1], [2,1], [3,2], [-1,-1], [-2,-1], [-3,-2]])

# Loading the PCA algorithm
pca_1 = PCA(n_components=1)

# Let's fit the data. We do not need to scale it, 
# since sklearn's implementation already handles it.
pca_1.fit(X)

# A float value that explains the variance from the original data
pca_1.explained_variance_ratio_ 

X_trans_1 = pca_1.transform(X)

X_reduced_1 = pca_1.inverse_transform(X_trans_1)
```
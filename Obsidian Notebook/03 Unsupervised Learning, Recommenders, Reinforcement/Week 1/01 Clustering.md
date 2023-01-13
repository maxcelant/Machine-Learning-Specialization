### Introduction
- Since the data is unlabeled, Clustering looks for one particular type of structure in the data
- Looks at data and sees if it can be grouped

### K-means Intuition
- Will repeatedly do 2 things:
	1. Assign points to cluster centroids
	2. Move cluster centroids
![[Pasted image 20221129165904.png|center]]
- Will look at each point and say if it is closer to the red cross or the blue cluster centroid
- Next, each centroid will look at all it's points and moves to the average location of all of the points
![[Pasted image 20221129170125.png|center]]
- Now, we assign the points again two the new placements of the centroids
![[Pasted image 20221129170233.png|center]]
- We repeat this until there are no more changes to the cluster centroids

### K-means Algorithm
- Randomly initialize `K` cluster centroids
- Each centroid is denoted `µ1, µ2,...µK`
- Each centroid is a n-dimensional vector, depending on how many features the model has
	- `µ1 = [x1,x2,x3,x4,x5]` in this case is a 5th dimensional vector
- But also note that every training example `x[1], x[2],...x[n]` also has the same amount of features as the centroids! In the example, they each have two features, hence the x,y plane
- If each example `x` is a vector of 5 numbers, then each cluster centroid `µK` is also going to be a vector of 5 numbers.
<u>**Step 1: Assing Points to Cluster Centroids**</u>
- `x[i]` is a training example
- `c[i]` is the index (from 1 to K) of cluster centroid closest to `x[i]`
![[Pasted image 20221129171416.png|center]]
- For example, if `x[12]` is closer to `µ2`, then `c[12] = 2`
<u>**Step 2: Move Cluster Centroids**</u>
- For each centroid, find the mean of points assigned to that cluster 
![[Pasted image 20221129171547.png|center]]
- For example: `µ2 = 1/4 * [x[1], x[5], x[6], x[10]]`
- Remember that each `x[1]...x[m]` has n-features!

### Optimization Objective
- Cost function is also called the **Distortion Algorithm**
- `c[i]` = index of cluster (1, 2, ... K) to which example `x[i]` is currently assigned or in other words, index of cluster centroid CLOSEST to `x[i]`
- `µ_c[i]` = cluster centroid of cluster to which example `x[i]` has been assigned 
	- For example: 
		- if `x[10]` is the training example
		- `c[10]` is the index value of the centroid `x[10]` was assigned
		- `µ_c[10]` is the location of the centroid which `x[10]` was assigned
```python
# calculate the total cost/error
cost = 0 # J(c, µ)

# for every training example 1 to m
for i in range(m):
	# distance between a training example and the centroid it's assigned to
	cost += abs(x[i] - µ_c[i])**2
cost /= m
```
![[Pasted image 20221129173827.png|center]]

### Initializing K-means
- Instead of choosing random locations for the centroids, we can choose to put them at the location of training examples! This will make the model run quicker and will produce better output
![[Pasted image 20221129180923.png|center]]
- We can do various iterations of this to see which input will produce the best clusters
![[Pasted image 20221129181041.png|center]]

```
For i = 1 to 100 {
	Randomly initialize K-means
	Run K-means. Get c[1]...c[m], µ[1]..µ[k]
	Compute cost funtion
}
Pick set of clusters that gave lowest cost J
```

### Choosing the Number of Clusters
- Often, you want to get clusters for some later purpose.
- Evaluate K-means based on how well it performs on that later purpose.

### Lab
**Finding the Closest Centroid to Each Training Example**
```python
def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): k centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    """
    
    K = centroids.shape[0]
	m = X.shape[0]
    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

	# For every training example
    for i in range(m):
        dist = [] # Distance between current example and every centroid
        for j in range(K):
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            dist.append(norm_ij)
        
        idx[i] = np.argmin(dist) # Return the minimum distance
    
    return idx
```

**Compute the Centroids**
```python
def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))

    for i in range(K):
        points = X[idx == i] # get all the points for the i-th centroid
        centroids[i] = np.mean(points, axis=0) # move it to the mean of its points
    
    return centroids
```


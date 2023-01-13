- Softmax regression algorithm is a generalization of logistic regression, which is a binary classification algorithm to the multiclass classification contexts.
- For each possible output, you calculate the CHANCE that its one category over all the others
![[Pasted image 20221112120023.png|center]]
- Notice how the total still adds up to 1


#### Generalize...
- If we have N possible outputs, then
![[Pasted image 20221112120129.png]]

#### In Code...
```python
def my_softmax(z):  
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """    
    ez = np.exp(z)
    a = ez / np.sum(ez)
    return a
```


#### Cost Function
- Recall that for logistic regression, this is the loss function and cost function

![[Pasted image 20221112120326.png|center]]

- In softmax, the loss function can be defined like so
![[Pasted image 20221112120612.png|center]]
- Note that depending on value of y, you only solve that loss function, not all of them

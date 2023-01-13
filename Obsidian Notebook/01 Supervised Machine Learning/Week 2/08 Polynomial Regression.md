- Sometimes a curve fits the data better than a straight line

![[Pasted image 20221105121409.png|center]]

- Using square root is also another option, if you need one that flattens out more towards the top

```py
x = np.arange(0,20)
y = x**2
X = np.c_[x, x**2, x**3] # create a new column for each

# using these engineered features, we can get a better fit for the model
```

![[Pasted image 20221105122949.png|center]]

- With this model, we would get values `[0.08, 0.54, 0.03]` and a `b = 0.0106`
- This means that the model after training is: ![[Pasted image 20221105123108.png]]

- This is how `np.c_` works

```py
x = np.arange(0, 20, 1)
print(x)

X = np.c_[x, x**2, x**3] 
print(X)
```
![[Pasted image 20221105123238.png|center]]
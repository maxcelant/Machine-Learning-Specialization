#### Using Entropy to measure Impurity
- Impurity can be summarized as not being entirely one class and being a mix of two or more classes, the more mixing their is, the more impure it is.
- `p1` = fraction of examples that are cats.
- Entropy denoted as `H(p1)`.
- if `p1 = 3/6` so 3 of the 6 elements are cats, then `H(p1) = 1`.
- **REMEMBER** the higher the entropy, the WORST it is, we are trying to get close to 0
![[Pasted image 20221127123315.png|center]]
- We have the highest 'impurity' when its a 50%, because its neither this or that, which makes it a bad way of splitting the data.
- if `p1 = 6/6` so 6 of the 6 elements are cats, then `H(p1) = 0`, which is GOOD! **Full purity**. 
![[Pasted image 20221127123524.png|center]]


### Calculating H(p1)
![[Pasted image 20230108120406.png]]

```python
p1 = 0.5 # Fraction of examples that are cats
p0 = 1 - p1
H_p1 = -p1 * np.log2(p1) - p0 * np.log2(p0)
```
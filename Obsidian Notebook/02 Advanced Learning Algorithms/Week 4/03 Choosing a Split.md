- **Information Gain** - The reduction of entropy
![[Pasted image 20221127143113.png|center]]
- We then need to calculate the average of the entropy
- `(percent / total) * H(p1)`
![[Pasted image 20221127143402.png|center]]
- We also need to subtract from the entropy at the root node `H(p1)`
![[Pasted image 20221127143611.png|center]]
- Since splitting by ear shape gives us the highest reduction entropy so we choose that
- `p1_left`/`p1_right` the number of elements that we want in the left/right sub-brach
- `w_left`/`w_right` the total number of elements in the left/right sub-branch divided by the total in the root
- `p1_root` number of elements we want in the root node
```python
H(p1_root) - (w_left * H(p1_left) + w_right * H(p1_right))
```
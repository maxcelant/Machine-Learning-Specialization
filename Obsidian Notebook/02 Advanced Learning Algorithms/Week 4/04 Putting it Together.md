1. Start with all features at root node
2. Calculate information gain for all possible features, and pick the one with highest information gain
3. Split dataset according to selected feature and create left and right branches of the tree
4. Keep on repeating this cycle
5. Stop when: 
	- Node is 100% one class
	- When splitting node causes you to exceed max depth
	- Information gain from additional splits is less than threshold

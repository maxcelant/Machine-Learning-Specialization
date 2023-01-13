### Introduction
![[Pasted image 20221127120452.png|center]]
- When you have a new example, it will start at the top and make its way down
- **Decision Nodes** are the intermediate nodes
- **Leaf Nodes** are the bottom or last nodes

### Learning Process
- **How do you choose which feature to split on at each node?**
	- Try to Maximize Purity (or Minimize Impurity)
![[Pasted image 20221127122613.png|center]]
- **When to stop splitting?**
	- When a node is 100% one class
	- When splitting a node will result in a tress exceeding a maximum depth
	- When improvements in a purity score are below a threshold
	- When number of examples in a node is below a threshold
		- *If there are only three items in a particular node, then its not worth splitting further*

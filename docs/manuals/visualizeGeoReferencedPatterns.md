[__<--__ Previous ](aboutPAMI.html)|[Home](index.html)|[_Next_-->](organization.html)

## Visualizing top-_k_ geo-referenced (or spatially) interesting patterns

In this page, we describe the methodology to view top-_k_ geo-referenced (or spatially) interesting patterns in a database.
The top-_k_ patterns were selected based on the number of items in a pattern.

__Note:__ This article assumes that the user has already mined and stored the interesting patterns in a geo-referenced database.

### Step 1: Specify the input file that contains the geo-referenced patterns
```Python
iFile = "interestingPatternsDiscovered.csv"
```

### Step 2: Specify the _k_ value as a parameter
```Python
k=5
```

### Step 3: Call the program

```Python
#import the program
from PAMI.extras.graph import visualizePatterns as visual

obj = visual.visualizePatterns(iFile, k)
obj.visualize()

```

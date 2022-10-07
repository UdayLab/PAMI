# Advanced Tutorial on Implementing ECLAT Algorithm

In this tutorial, we will discuss the second approach to find frequent patterns in big data using ECLAT algorithm.

[__Advanced approach:__](#advApproach) Here, we generalize the basic approach by presenting the steps to discover frequent patterns using multiple minimum support values.

***

#### In this tutorial, we explain how the ECLAT algorithm  can be implemented by varying the minimum support values

#### Step 1: Import the ECLAT algorithm and pandas data frame


```python
from PAMI.frequentPattern.basic import ECLAT  as alg
import pandas as pd
```

#### Step 2: Specify the following input parameters


```python
inputFile = '/userData/likhitha/new/frequentPattern/transactional_T10I4D100K.csv'
seperator='\t'
minimumSupportCountList = [100, 150, 200, 250, 300] 
#minimumSupport can also specified between 0 to 1. E.g., minSupList = [0.005, 0.006, 0.007, 0.008, 0.009]

result = pd.DataFrame(columns=['algorithm', 'minSup', 'patterns', 'runtime', 'memory']) 
#initialize a data frame to store the results of ECLAT algorithm
```

#### Step 3: Execute the ECLAT algorithm using a for loop


```python
algorithm = 'ECLAT'  #specify the algorithm name
for minSupCount in minimumSupportCountList:
    obj = alg.ECLAT(inputFile, minSup=minSupCount, sep=seperator)
    obj.startMine()
    #store the results in the data frame
    result.loc[result.shape[0]] = [algorithm, minSupCount, len(obj.getPatterns()), obj.getRuntime(), obj.getMemoryRSS()]

```

    Frequent patterns were generated successfully using ECLAT algorithm
    Frequent patterns were generated successfully using ECLAT algorithm
    Frequent patterns were generated successfully using ECLAT algorithm
    Frequent patterns were generated successfully using ECLAT algorithm
    Frequent patterns were generated successfully using ECLAT algorithm



```python
print(result)
```

#### Step 5: Visualizing the results

##### Step 5.1 Importing the plot library


```python
from PAMI.extras.graph import plotLineGraphsFromDataFrame as plt
```

##### Step 5.2. Plotting the number of patterns


```python
ab = plt.plotGraphsFromDataFrame(result)
ab.plotGraphsFromDataFrame() #drawPlots()
```

### Step 6: Saving the results as latex files


```python
from PAMI.extras.graph import generateLatexFileFromDataFrame as gdf
gdf.generateLatexCode(result)
```


```python

```

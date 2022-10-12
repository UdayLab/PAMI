# Advanced Tutorial on Implementing PFECLAT Algorithm

In this tutorial, we will discuss the second approach to find Periodic Frequent patterns in big data using top algorithm.

[__Advanced approach:__](#advApproach) Here, we generalize the basic approach by presenting the steps to discover Periodic Frequent patterns using multiple minimum support values.

***

#### In this tutorial, we explain how the Periodic Frequent ECLAT (PFECLAT) algorithm  can be implemented by varying the minimum support values

#### Step 1: Import the PFECLAT algorithm and pandas data frame


```python
from PAMI.periodicFrequentPattern.basic import PFECLAT  as alg
import pandas as pd
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'temporal_T10I4D100K.csv'
seperator = '\t'
maxmunPeriodCount = 5000
minimumSupportCountList = [100, 150, 200, 250, 300] 
#minimumSupport can also specified between 0 to 1. E.g., minSupList = [0.005, 0.006, 0.007, 0.008, 0.009]

result = pd.DataFrame(columns=['algorithm', 'minSup', 'maxPer','patterns', 'runtime', 'memory']) 
#initialize a data frame to store the results of PFECLAT algorithm
```

#### Step 3: Execute the PFECLAT algorithm using a for loop


```python
algorithm = 'PFECLAT'  #specify the algorithm name
for minSupCount in minimumSupportCountList:
    obj = alg.PFECLAT('temporal_T10I4D100K.csv', minSup=minSupCount,maxPer=maxmunPeriodCount, sep=seperator)
    obj.startMine()
    #store the results in the data frame
    result.loc[result.shape[0]] = [algorithm, minSupCount,maxmunPeriodCount, len(obj.getPatterns()), obj.getRuntime(), obj.getMemoryRSS()]

```

    Periodic-Frequent patterns were generated successfully using PFECLAT algorithm 
    Periodic-Frequent patterns were generated successfully using PFECLAT algorithm 
    Periodic-Frequent patterns were generated successfully using PFECLAT algorithm 
    Periodic-Frequent patterns were generated successfully using PFECLAT algorithm 
    Periodic-Frequent patterns were generated successfully using PFECLAT algorithm 



```python
print(result)
```

      algorithm  minSup  maxPer  patterns    runtime     memory
    0   PFECLAT     100    5000     25462  45.491256  487788544
    1   PFECLAT     150    5000     18982  28.728825  439586816
    2   PFECLAT     200    5000     13251  18.953881  435798016
    3   PFECLAT     250    5000      7702  13.450154  436830208
    4   PFECLAT     300    5000      4552  10.846030  437895168


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


    
![png](output_15_0.png)
    


    Graph for No Of Patterns is successfully generated!



    
![png](output_15_2.png)
    


    Graph for Runtime taken is successfully generated!



    
![png](output_15_4.png)
    


    Graph for memory consumption is successfully generated!


### Step 6: Saving the results as latex files


```python
from PAMI.extras.graph import generateLatexFileFromDataFrame as gdf
gdf.generateLatexCode(result)
```

    Latex files generated successfully


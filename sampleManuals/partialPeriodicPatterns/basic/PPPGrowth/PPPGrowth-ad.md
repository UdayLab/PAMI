# Advanced Tutorial on Implementing PPPGrowth Algorithm

In this tutorial, we will discuss the second approach to find Partial Periodic Pattern in big data using PPPGrowth (3PGrowth) algorithm.

[__Advanced approach:__](#advApproach) Here, we generalize the basic approach by presenting the steps to discover Partial Periodic Pattern using multiple minimum support values.

***

#### In this tutorial, we explain how the PPPGrowth algorithm  can be implemented by varying the minimum support values

#### Step 1: Import the PPPGrowth algorithm and pandas data frame


```python
from PAMI.partialPeriodicPattern.basic import PPPGrowth  as alg
import pandas as pd
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'temporal_T10I4D100K.csv'
seperator = '\t'
periodCount = 5000
periodicSupportCountList = [100, 150, 200, 250, 300] 
#minimumSupport can also specified between 0 to 1. E.g., minSupList = [0.005, 0.006, 0.007, 0.008, 0.009]

result = pd.DataFrame(columns=['algorithm', 'minSup', 'period','patterns', 'runtime', 'memory']) 
#initialize a data frame to store the results of PPPGrowth algorithm
```

#### Step 3: Execute the PPPGrowth algorithm using a for loop


```python
algorithm = 'PPPGrowth'  #specify the algorithm name
for periodicSupportCount in periodicSupportCountList:
    obj = alg.PPPGrowth('temporal_T10I4D100K.csv', periodicSupport=periodicSupportCount, period=periodCount, sep=seperator)
    obj.startMine()
    #store the results in the data frame
    result.loc[result.shape[0]] = [algorithm, periodicSupportCount,periodCount, len(obj.getPatterns()), obj.getRuntime(), obj.getMemoryRSS()]
```

    Partial Periodic Patterns were generated successfully using 3PGrowth algorithm 
    Partial Periodic Patterns were generated successfully using 3PGrowth algorithm 
    Partial Periodic Patterns were generated successfully using 3PGrowth algorithm 
    Partial Periodic Patterns were generated successfully using 3PGrowth algorithm 
    Partial Periodic Patterns were generated successfully using 3PGrowth algorithm 


#### Step 4: Print the result


```python
print(result)
```

       algorithm  minSup  period  patterns    runtime     memory
    0  PPPGrowth     100    5000     27162  17.409416  577286144
    1  PPPGrowth     150    5000     18977  16.247539  574623744
    2  PPPGrowth     200    5000     13150  15.139839  572116992
    3  PPPGrowth     250    5000      7627  14.257152  568352768
    4  PPPGrowth     300    5000      4506  12.911864  565395456


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


    
![png](output_16_0.png)
    


    Graph for No Of Patterns is successfully generated!



    
![png](output_16_2.png)
    


    Graph for Runtime taken is successfully generated!



    
![png](output_16_4.png)
    


    Graph for memory consumption is successfully generated!


### Step 6: Saving the results as latex files

```python
from PAMI.extras.graph import DF2Tex as gdf

gdf.generateLatexCode(result)
```

    Latex files generated successfully


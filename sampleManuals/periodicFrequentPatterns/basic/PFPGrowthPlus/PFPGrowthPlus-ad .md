# Advanced Tutorial on Implementing PFPGrowthPlus Algorithm

In this tutorial, we will discuss the second approaches to find Periodic Frequent patterns in big data using PFPGrowth++ algorithm.

[__Advanced approach:__](#advApproach) Here, we generalize the basic approach by presenting the steps to discover Periodic Frequent patterns using multiple minimum support values.

***

#### In this tutorial, we explain how the Periodic Frequent Puttern Growth Plus (PFPGrowthPlus) algorithm  can be implemented by varying the minimum support values

#### Step 1: Import the PFPGrowthPlus algorithm and pandas data frame


```python
from PAMI.periodicFrequentPattern.basic import PFPGrowthPlus  as alg
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
#initialize a data frame to store the results of PFPGrowthPlus algorithm
```

#### Step 3: Execute the PFPGrowthPlus algorithm using a for loop


```python
algorithm = 'PFPGrowthPlus'  #specify the algorithm name
for minSupCount in minimumSupportCountList:
    obj = alg.PFPGrowthPlus('temporal_T10I4D100K.csv', minSup=minSupCount,maxPer=maxmunPeriodCount, sep=seperator)
    obj.startMine()
    #store the results in the data frame
    result.loc[result.shape[0]] = [algorithm, minSupCount,maxmunPeriodCount, len(obj.getPatterns()), obj.getRuntime(), obj.getMemoryRSS()]

```

    periodic-frequent patterns were generated successfully using PFPGrowth++ algorithm 
    periodic-frequent patterns were generated successfully using PFPGrowth++ algorithm 
    periodic-frequent patterns were generated successfully using PFPGrowth++ algorithm 
    periodic-frequent patterns were generated successfully using PFPGrowth++ algorithm 
    periodic-frequent patterns were generated successfully using PFPGrowth++ algorithm 



```python
print(result)
```

           algorithm  minSup  maxPer  patterns    runtime     memory
    0  PFPGrowthPlus     100    5000     25462  12.942107  579301376
    1  PFPGrowthPlus     150    5000     18982  11.716884  577437696
    2  PFPGrowthPlus     200    5000     13251  11.130969  574922752
    3  PFPGrowthPlus     250    5000      7702  10.318596  570114048
    4  PFPGrowthPlus     300    5000      4552  10.184319  566091776


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
from PAMI.extras.graph import DF2Tex as gdf

gdf.generateLatexCode(result)
```

    Latex files generated successfully


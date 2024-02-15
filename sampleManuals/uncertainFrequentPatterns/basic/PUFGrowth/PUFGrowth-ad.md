# Advanced Tutorial on Implementing PUFGrowth Algorithm

***

#### In this tutorial, we explain how the PUFGrowth algorithm  can be implemented by varying the minimum support values

#### Step 1: Import the PUFGrowth algorithm and pandas data frame


```python
from PAMI.uncertainFrequentPattern.basic import PUFGrowth  as alg
import pandas as pd
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'uncertainTransaction_T10I4D200K.csv'
seperator = '\t'
minimumSupportCountList = [300, 400, 500, 600, 700] 
#minimumSupport can also specified between 0 to 1. E.g., minSupList = [0.005, 0.006, 0.007, 0.008, 0.009]
result = pd.DataFrame(columns=['algorithm', 'minSup', 'patterns', 'runtime', 'memory']) 
#initialize a data frame to store the results of PUFGrowth algorithm
```

#### Step 3: Execute the PUFGrowth algorithm using a for loop


```python
algorithm = 'PUFGrowth'  #specify the algorithm name
for minSupCount in minimumSupportCountList:
    obj = alg.PUFGrowth(inputFile, minSup=minSupCount, sep=seperator)
    obj.startMine()
    #store the results in the data frame
    result.loc[result.shape[0]] = [algorithm, minSupCount, len(obj.getPatterns()), obj.getRuntime(), obj.getMemoryRSS()]

```

    Total number of false patterns generated: 2007
    Uncertain Frequent patterns were generated successfully using PUFGrowth algorithm
    Total number of false patterns generated: 2221
    Uncertain Frequent patterns were generated successfully using PUFGrowth algorithm
    Total number of false patterns generated: 2316
    Uncertain Frequent patterns were generated successfully using PUFGrowth algorithm
    Total number of false patterns generated: 2377
    Uncertain Frequent patterns were generated successfully using PUFGrowth algorithm
    Total number of false patterns generated: 2419
    Uncertain Frequent patterns were generated successfully using PUFGrowth algorithm



```python
print(result)
```

       algorithm  minSup  patterns     runtime      memory
    0  PUFGrowth     300       874  298.410130  1004875776
    1  PUFGrowth     400       674  297.416058   995827712
    2  PUFGrowth     500       579  298.086999   984948736
    3  PUFGrowth     600       518  298.529948   972185600
    4  PUFGrowth     700       476  295.230906   961323008


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


    
![png](output_14_0.png)
    


    Graph for No Of Patterns is successfully generated!



    
![png](output_14_2.png)
    


    Graph for Runtime taken is successfully generated!



    
![png](output_14_4.png)
    


    Graph for memory consumption is successfully generated!


### Step 6: Saving the results as latex files

```python
from PAMI.extras.graph import DF2Tex as gdf

gdf.generateLatexCode(result)
```

    Latex files generated successfully



```python

```

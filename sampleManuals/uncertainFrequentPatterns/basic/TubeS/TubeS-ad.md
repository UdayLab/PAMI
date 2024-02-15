# Advanced Tutorial on Implementing TubeS Algorithm

***

#### In this tutorial, we explain how the TubeS algorithm  can be implemented by varying the minimum support values

#### Step 1: Import the TubeS algorithm and pandas data frame


```python
import TubeS  as alg
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

#### Step 3: Execute the TubeS algorithm using a for loop


```python
algorithm = 'TubeS'  #specify the algorithm name
for minSupCount in minimumSupportCountList:
    obj = alg.TubeS(inputFile, minSup=minSupCount, sep=seperator)
    obj.startMine()
    #store the results in the data frame
    result.loc[result.shape[0]] = [algorithm, minSupCount, len(obj.getPatterns()), obj.getRuntime(), obj.getMemoryRSS()]

```

    Total number of false patterns generated: 1969
    Uncertain Frequent patterns were generated successfully using TubeS algorithm
    Total number of false patterns generated: 2169
    Uncertain Frequent patterns were generated successfully using TubeS algorithm
    Total number of false patterns generated: 2264
    Uncertain Frequent patterns were generated successfully using TubeS algorithm
    Total number of false patterns generated: 2325
    Uncertain Frequent patterns were generated successfully using TubeS algorithm
    Total number of false patterns generated: 2367
    Uncertain Frequent patterns were generated successfully using TubeS algorithm



```python
print(result)
```

      algorithm  minSup  patterns     runtime      memory
    0     TubeS     300       874  299.080104  1207087104
    1     TubeS     400       674  296.779736  1191620608
    2     TubeS     500       579  294.481138  1173700608
    3     TubeS     600       518  296.047587  1152577536
    4     TubeS     700       476  297.011776  1134804992


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

# Discovering High Utility Spatial Pattern in Big Data Using HDSHUIM Algorithm

In this tutorial, we will discuss the first approach to find High Utility Spatial Pattern in big data using HDSHUIM algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover High Utility Spatial Pattern using a single minimum utility value.

***

## <a id='basicApproach'>Basic approach: Executing HDSHUIM on a single dataset at a particular minimum utility value</a>

#### Step 1: Import the HDSHUIM algorithm


```python
from PAMI.highUtilitySpatialPattern.basic import HDSHUIM  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'utility_mushroom.txt'

neighborFile = 'mushroom_neighbourhoodFile_9.txt'
minUtilCount = 5000
seperator = ' '       
```

#### Step 3: Execute the HDSHUIM algorithm


```python
obj = alg.HDSHUIM(iFile=inputFile, nFile=neighborFile, minUtil=minUtilCount,  sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='spatialUtiltiyPatternsMinUtil5000.txt')
```

##### Step 4.2. Storing the generated patterns in a data frame


```python
frequentPatternsDF= obj.getPatternsAsDataFrame()
```

#### Step 5: Getting the statistics

##### Step 5.1: Total number of discovered patterns 


```python
print('Total No of patterns: ' + str(len(frequentPatternsDF)))
```

    Total No of patterns: 128


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 1.7327630519866943


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 170811392
    Memory (USS): 132603904


# Discovering High Utility Frequent Spatial Pattern in Big Data Using SHUFIM Algorithm

In this tutorial, we will discuss two approaches to find High Utility Frequent Spatial Pattern in big data using SHUFIM algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover High Utility Frequent Spatial Pattern using a single minimum utility value

***

## <a id='basicApproach'>Basic approach: Executing SHUFIM on a single dataset at a particular minimum utility value</a>

#### Step 1: Import the SHUFIM algorithm

```python
from PAMI.highUtilityGeoreferencedFrequentPattern.basic import SHUFIM as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'utility_mushroom.txt'
neighborFile = 'mushroom_neighbourhood.txt'
minUtilCount = 10000
minSup = 100
seperator = ' '       
```

#### Step 3: Execute the SHUFIM algorithm


```python
obj = alg.SHUFIM(iFile=inputFile, nFile=neighborFile, minUtil=minUtilCount, minSup=minSup, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Spatial High Utility Frequent Itemsets generated successfully using SHUFIM algorithm


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='spatialUtilityFrequentPatterns.txt')
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

    Total No of patterns: 102


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 7.566678285598755


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 151867392
    Memory (USS): 113205248


# Discovering Fuzzy Frequent Spatial Pattern in Big Data Using FFSPMiner Algorithm

In this tutorial, we will discuss the first approach to find Fuzzy Frequent Spatial Pattern in big data using FFSPMiner algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover Fuzzy Frequent Spatial Pattern using a single minimum support value


***

## <a id='basicApproach'>Basic approach: Executing FFSPMiner on a single dataset at a particular minimum support value</a>

#### Step 1: Import the FFSPMiner algorithm


```python
from PAMI.fuzzyFrequentSpatialPattern.basic import FFSPMiner  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'fuzzyTransactional_T10I4D100K.csv'

minimumSupportCount=100  #Users can also specify this constraint between 0 to 1.
neighborFile='fuzzy_T10I4D100K_neighbour.csv'
seperator=' '       
```

#### Step 3: Execute the FFSPMiner algorithm


```python
obj = alg.FFSPMiner(iFile=inputFile, nFile=neighborFile, minSup=minimumSupportCount,sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='frequentPatternsMinSupCount1000.txt')
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

    Total No of patterns: 797


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 7.619616985321045


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 457887744
    Memory (USS): 419237888


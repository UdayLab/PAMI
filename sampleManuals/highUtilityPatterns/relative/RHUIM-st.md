# Discovering Relative High Utility patterns in Big Data Using RHUIM Algorithm

In this tutorial, we will discuss the first approach to find Relative High Utility patterns in big data using RHUIM algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover Relative High Utility patterns using a single minimum utility value and single minimum utility ratio.

***

## <a id='basicApproach'>Basic approach: Executing RHUIM on a single dataset at a particular minimum utility value and minimum utility ratio</a>

#### Step 1: Import the RHUIM algorithm


```python
from PAMI.relativeHighUtilityPatterns.basic import RHUIM as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'Utility_T10I4D100K.csv'
  #Users can also specify this constraint between 0 to 1.
minUtilCount=50000
minUtilRatio=0.6
seperator='\t'       
```

#### Step 3: Execute the RHUIM algorithm


```python
obj = alg.RHUIM(iFile=inputFile,minUtil=minUtilCount, minUR=minUtilRatio,  sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Relative High Utility patterns were generated successfully using RHUIM algorithm


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='relativeHighUtilityPatternsMinUtil30000.txt')
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

    Total No of patterns: 5968


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 109.58998990058899


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 208748544
    Memory (USS): 170840064


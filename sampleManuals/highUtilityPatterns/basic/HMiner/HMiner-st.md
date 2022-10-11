# Discovering High Utility patterns in Big Data Using HMiner Algorithm

In this tutorial, we will discuss the first approach to find High Utility patterns in big data using HMiner algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover High Utility patterns using a single minimum utility value

***

## <a id='basicApproach'>Basic approach: Executing HMiner on a single dataset at a particular minimum utility value</a>

#### Step 1: Import the HMiner algorithm


```python
from PAMI.highUtilityPatterns.basic import HMiner  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'Utility_T10I4D100K.csv'
  #Users can also specify this constraint between 0 to 1.
minUtilCount=30000
seperator='\t'       
```

#### Step 3: Execute the HMiner algorithm


```python
obj = alg.HMiner(iFile1=inputFile,minUtil=minUtilCount,  sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    High Utility patterns were generated successfully using HMiner algorithm


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='frequentPatternsMinSupCount100.txt')
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

    Total No of patterns: 14468


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 144.15201711654663


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 528842752
    Memory (USS): 490536960


# Discovering Weighted Frequent Patterns in Big Data Using WFIM Algorithm

In this tutorial, we will discuss two approaches to find weighted frequent patterns in big data using WFIM algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover weighted frequent patterns using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing WFIM on a single dataset at a particular minimum support value</a>

#### Step 1: Import the WFIM algorithm


```python
from PAMI.weightedFrequentPattern.basic import WFIM as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'Transactional_T10I4D100K.csv'
weightFile = 'T10_weights.txt'
minimumSupportCount = 500  #Users can also specify this constraint between 0 to 1.
minWeight = 50
seperator='\t'       
```

#### Step 3: Execute the WFIM algorithm


```python
obj = alg.WFIM(iFile=inputFile, wFile=weightFile, minSup=minimumSupportCount, minWeight=minWeight, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Weighted Frequent patterns were generated successfully using WFIM algorithm


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save('weightedFrequentPatternsMinSupCount500.txt')
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

    Total No of patterns: 1066


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 8.086735010147095


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 472535040
    Memory (USS): 434274304


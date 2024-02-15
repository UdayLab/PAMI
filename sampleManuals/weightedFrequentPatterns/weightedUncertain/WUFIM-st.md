# Discovering Weighted Frequent Patterns in Uncertain Big Data Using WUFIM Algorithm

In this tutorial, we will discuss two approaches to find frequent patterns in uncertain big data using WUFIM algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover frequent patterns using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing WUFIM on a single dataset at a particular minimum support value</a>

#### Step 1: Import the WUFIM algorithm


```python
from PAMI.weightedUncertainFrequentPattern.basic import WUFIM as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'uncertainTransaction_T10I4D200K.csv'
weightFile = 'T10_weights.txt'
minimumSupportCount = 500  #Users can also specify this constraint between 0 to 1.
seperator = '\t'       
```

#### Step 3: Execute the WUFIM algorithm


```python
obj = alg.WUFIM(iFile=inputFile, wFile=weightFile, expSup=500, expWSup=300, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Weighted Frequent patterns were generated  successfully using WUFIM algorithm


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

    Total No of patterns: 834


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 11.865854024887085


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 980291584
    Memory (USS): 941871104


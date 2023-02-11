# Discovering Weighted Frequent Regular Patterns in Big Data Using WFRIM Algorithm

In this tutorial, we will discuss two approaches to find weighted frequent regular patterns in big data using WFRIM algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover weighted frequent regular patterns using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing WFRIM on a single dataset at a particular minimum support value</a>

#### Step 1: Import the WFRIM algorithm


```python
from PAMI.weightedFrequentRegularPattern.basic import WFRIMiner as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'temporal_T10I4D100K.csv'
weightFile = 'T10_weights.txt'
minimumSupportCount = 500  #Users can also specify this constraint between 0 to 1.
regularity=3000
seperator='\t'       
```

#### Step 3: Execute the WFRIM algorithm


```python
obj = alg.WFRIMiner(iFile=inputFile, _wFile=weightFile, WS=minimumSupportCount, regularity=regularity,  sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Weighted Frequent Regular patterns were generated successfully using WFRIM algorithm


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save('weightedRegularPatternsMinSupCount500.txt')
```

##### Step 4.2. Storing the generated patterns in a data frame


```python
weightedRegularPatternsDF= obj.getPatternsAsDataFrame()
```

#### Step 5: Getting the statistics

##### Step 5.1: Total number of discovered patterns 


```python
print('Total No of patterns: ' + str(len(weightedRegularPatternsDF)))
```

    Total No of patterns: 1202


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 13.080559253692627


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 568975360
    Memory (USS): 530284544


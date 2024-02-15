# Discovering Top-K Periodic Frequent patterns in Big Data Using k-PFPMiner Algorithm

In this tutorial, we will discuss the first approach to find Top-K Periodic Frequent patterns in big data using kPFPMiner algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover Top-K Periodic Frequent patterns using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing  kPFPMiner on a single dataset at a particular k value</a>

#### Step 1: Import the kPFPMiner algorithm


```python
from PAMI.periodicFrequentPattern.topk.kPFPMiner import kPFPMiner as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'temporal_T10I4D100K.csv'

k = 100  #Users can also specify this constraint between 0 to 1.
seperator = '\t'       
```

#### Step 3: Execute the PFECLAT algorithm


```python
obj = alg.kPFPMiner(iFile=inputFile, k=k, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    kPFPMiner has successfully generated top-k frequent patterns


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='topKperiodicFrequentPatterns100.txt')
```

##### Step 4.2. Storing the generated patterns in a data frame


```python
periodicFrequentPatternsDF= obj.getPatternsAsDataFrame()
```

#### Step 5: Getting the statistics

##### Step 5.1: Total number of discovered patterns 


```python
print('Total No of patterns: ' + str(len(periodicFrequentPatternsDF)))
```

    Total No of patterns: 100


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 2.3006834983825684


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 222756864
    Memory (USS): 184029184


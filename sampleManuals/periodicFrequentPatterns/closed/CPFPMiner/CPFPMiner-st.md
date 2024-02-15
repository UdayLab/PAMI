# Discovering Closed periodic frequent patterns in Big Data Using CPFPMiner Algorithm

In this tutorial, we will discuss the first approach to find Closed periodic frequent patterns in big data using CPFPMiner algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover Closed periodic frequent patterns using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing CPFPMiner on a single dataset at a particular minimum support value</a>

#### Step 1: Import the CPFPMiner algorithm


```python
from PAMI.periodicFrequentPattern.closed import CPFPMiner  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'temporal_T10I4D100K.csv'

minimumSupportCount = 100  #Users can also specify this constraint between 0 to 1.
maxmunPeriodCount = 5000
seperator = '\t'       
```

#### Step 3: Execute the CPFPMiner algorithm


```python
obj = alg.CPFPMiner(iFile=inputFile, minSup=minimumSupportCount,maxPer=maxmunPeriodCount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Closed periodic frequent patterns were generated successfully using CPFPMiner algorithm 


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='closedPeriodicFrequentPatternsMinSupCount100.txt')
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

    Total No of patterns: 24711


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 25.700572967529297


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 144171008
    Memory (USS): 105672704


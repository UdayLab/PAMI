# Discovering Local Periodic patterns in Big Data Using LPPMBreadth Algorithm

In this tutorial, we will discuss the first approach to find Local Periodic patterns in big data using LPPMBreadth algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover Local Periodic patterns using particular maximum periodicity value, maximum spil over period value and minimum duration value

***

## <a id='basicApproach'>Basic approach: Executing LPPMBreadth on a single dataset at particular maximum periodicity value, maximum spil over period value and minimum duration value</a>

#### Step 1: Import the LPPMBreadth algorithm


```python
from PAMI.localPeriodicPattern.basic import LPPMBreadth as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'temporal_T10I4D100K.csv'

maximumPeriodicity = 50 #Users can also specify this constraint between 0 to 1.
maximumSpilOverPeriod = 50
minimumDuration = 500
seperator = ','       
```

#### Step 3: Execute the LPPMBreadth algorithm


```python
obj = alg.LPPMBreadth(iFile=inputFile, maxPer=maximumPeriodicity, maxSoPer=maximumSpilOverPeriod, minDur=minimumDuration, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='localPeriodicPattern_50_50_500.txt')
```

##### Step 4.2. Storing the generated patterns in a data frame


```python
localPeriodicPatternsDF= obj.getPatternsAsDataFrame()
```

#### Step 5: Getting the statistics

##### Step 5.1: Total number of discovered patterns 


```python
print('Total No of patterns: ' + str(len(localPeriodicPatternsDF)))
```

    Total No of patterns: 342


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 371.7866861820221


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 229715968
    Memory (USS): 191614976


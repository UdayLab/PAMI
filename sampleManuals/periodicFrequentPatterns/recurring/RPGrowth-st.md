# Discovering Recurring patterns in Big Data Using RPGrowth Algorithm

In this tutorial, we will discuss the first approach to find Recurring patterns in big data using RPGrowth algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover Recurring patterns using a single minimum support, maximum period, minimum recurrence values


***

## <a id='basicApproach'>Basic approach: Executing RPGrowth on a single dataset</a>

#### Step 1: Import the RPGrowth algorithm


```python
from PAMI.recurringPattern.basic import RPGrowth as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'temporal_T10I4D100K.csv'

minimumSupportCount = 100  #Users can also specify this constraint between 0 to 1.
maxmunPeriodCount = 5000
minRec = 1.8
seperator = '\t'       
```

#### Step 3: Execute the RPGrowth algorithm


```python
obj = alg.RPGrowth(iFile=inputFile, minPS=minimumSupportCount, maxPer=maxmunPeriodCount, minRec=minRec, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Recurring patterns were generated successfully using RPGrowth algorithm 


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='recurringPatternsMinSupCount100.txt')
```

##### Step 4.2. Storing the generated patterns in a data frame


```python
recurringPatternsDF= obj.getPatternsAsDataFrame()
```

#### Step 5: Getting the statistics

##### Step 5.1: Total number of discovered patterns 


```python
print('Total No of patterns: ' + str(len(recurringPatternsDF)))
```

    Total No of patterns: 25965


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 13.126968145370483


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 582684672
    Memory (USS): 544538624


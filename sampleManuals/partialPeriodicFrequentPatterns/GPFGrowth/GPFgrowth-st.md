# Discovering Partial Periodic Frequent Pattern in Big Data Using GPFgrowth Algorithm

In this tutorial, we will discuss the first approach to find Partial Periodic Frequent Pattern in big data using GPFgrowth algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover Partial Periodic Frequent Pattern using a single minimum support value


***

## <a id='basicApproach'>Basic approach: Executing GPFgrowth on a single dataset at a particular minimum support value</a>

#### Step 1: Import the GPFgrowth algorithm


```python
from PAMI.partialPeriodicFrequentPattern.basic import GPFgrowth  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'temporal_T10I4D100K.csv'
maxPerCount = 500
minPRcount = 0.5
minimumSupportCount = 100  #Users can also specify this constraint between 0 to 1.

seperator='\t'       
```

#### Step 3: Execute the GPFgrowth algorithm


```python
obj = alg.GPFgrowth(iFile=inputFile, minSup=minimumSupportCount,maxPer=maxPerCount,minPR=minPRcount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='partialPeriodicFrequentPatternsMinSupCount100.txt')
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

    Total No of patterns: 20688


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 54.974111795425415


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 696582144
    Memory (USS): 658100224


# Discovering Periodic-Frequent patterns in Big Data Using PFPMC Algorithm

In this tutorial, we will discuss the first approach to find Periodic-Frequent patterns in big data using PFPMC algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover Periodic-Frequent patterns using a single minimum support value


***

## <a id='basicApproach'>Basic approach: Executing PFPMC on a single dataset at a particular minimum support value</a>

#### Step 1: Import the PFPMC algorithm


```python
from PAMI.periodicFrequentPattern.basic import PFPMC  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'temporal_T10I4D100K.csv'

minimumSupportCount = 100  #Users can also specify this constraint between 0 to 1.
maxmunPeriodCount = 500
seperator = '\t'       
```

#### Step 3: Execute the PFPMC algorithm


```python
obj = alg.PFPMC(iFile=inputFile, minSup=minimumSupportCount,maxPer=maxmunPeriodCount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.savePatterns(outFile='periodicFrequentPatternsMinSupCount100.txt')
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

##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

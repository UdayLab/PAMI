# Discovering Periodic Frequent patterns in Big Data Using EPCPGrowth Algorithm

In this tutorial, we will discuss the first approach to find Correlated Periodic Frequent patterns in big data using EPCPGrowth algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover Periodic Frequent patterns using a single minimum support value


***

## <a id='basicApproach'>Basic approach: Executing EPCPGrowth on a single dataset at a particular minimum support value</a>

#### Step 1: Import the EPCPGrowth algorithm


```python
from PAMI.periodicCorrelatedPattern.basic import EPCPGrowth  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'temporal_T10I4D100K.csv'

minimumSupportCount = 100  #Users can also specify this constraint between 0 to 1.
maximumPeriodCount = 8000
minAllConfCount = 0.5
maxPerAllmaxPerConfCount = 4.0
seperator = '\t'       
```

#### Step 3: Execute the EPCPGrowth algorithm


```python
obj = alg.EPCPGrowth(iFile=inputFile, minSup=minimumSupportCount,minAllConf=minAllConfCount,maxPer=maximumPeriodCount,maxPerAllConf=maxPerAllmaxPerConfCount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Correlated Periodic-Frequent patterns were generated successfully using EPCPGrowth algorithm 


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='periodicFrequentPatternsMinSupCount100.txt')
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

    Total No of patterns: 843


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 15.619414567947388


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 573644800
    Memory (USS): 535248896



```python

```

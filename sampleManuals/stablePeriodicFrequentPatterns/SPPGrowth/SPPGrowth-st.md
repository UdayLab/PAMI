# Discovering Stable Periodic Frequent patterns in Big Data Using SPPGrowth Algorithm

In this tutorial, we will discuss the first approach to find Stable Periodic Frequent patterns in big data using SPPGrowth algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover Stable Periodic Frequent patterns using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing SPPGrowth on a single dataset at a particular minimum support value</a>

#### Step 1: Import the SPPGrowth algorithm


```python
from PAMI.stablePeriodicFrequentPattern.basic import SPPGrowth as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'temporal_T10I4D100K.csv'

minimumSupportCount = 100  #Users can also specify this constraint between 0 to 1.
maxmunPeriodCount = 5000
maxLaValue = 1000

seperator = '\t'       
```

#### Step 3: Execute the SPPGrowth algorithm


```python
obj = alg.SPPGrowth(inputFile, minimumSupportCount, maxmunPeriodCount, maxLaValue, seperator)    #initialize
obj.startMine()            #Start the mining process
```

    100 5000 1000
    Stable Periodic Frequent patterns were generated successfully using SPPGrowth algorithm 


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='stablePeriodicFrequentPatternsMinSupCount100.txt')
```

##### Step 4.2. Storing the generated patterns in a data frame


```python
stablePeriodicFrequentPatternsDF= obj.getPatternsAsDataFrame()
```

#### Step 5: Getting the statistics

##### Step 5.1: Total number of discovered patterns 


```python
print('Total No of patterns: ' + str(len(stablePeriodicFrequentPatternsDF)))
```

    Total No of patterns: 26974


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 14.430559158325195


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 578158592
    Memory (USS): 539422720


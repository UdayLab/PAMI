# Discovering Partial Periodic Pattern in Big Data Using PPPGrowth Algorithm

In this tutorial, we will discuss the first approach to find Partial Periodic Pattern in big data using PPPGrowth (3PGrowth) algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover Partial Periodic Pattern using a single minimum support value


***

## <a id='basicApproach'>Basic approach: Executing PPPGrowth on a single dataset at a particular minimum support value</a>

#### Step 1: Import the PPPGrowth algorithm


```python
from PAMI.partialPeriodicPattern.basic import PPPGrowth  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'temporal_T10I4D100K.csv'
periodCount = 5000
periodicSupportCount = 100  #Users can also specify this constraint between 0 to 1.

seperator='\t'       
```

#### Step 3: Execute the PPPGrowth algorithm


```python
obj = alg.PPPGrowth(iFile=inputFile,periodicSupport=periodicSupportCount, period=periodCount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Partial Periodic Patterns were generated successfully using 3PGrowth algorithm 


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='partialPeriodicPatternsMinSupCount1000.txt')
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

    Total No of patterns: 27162


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 13.475618839263916


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 576389120
    Memory (USS): 538152960


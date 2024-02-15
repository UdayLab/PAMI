# Discovering Partial Periodic Pattern in Big Data Using GThreePGrowth Algorithm

In this tutorial, we will discuss the first approach to find Partial Periodic Pattern in big data using GThreePGrowth (G3PGrowth) algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover Partial Periodic Pattern using minimum periodic support (minPS) and minimum relative periodic support (minRPS)


***

## <a id='basicApproach'>Basic approach: Executing GThreePGrowth on a single dataset at a particular minimum periodic support value and minimum relative periodic support </a>

#### Step 1: Import the GThreePGrowth algorithm


```python
from PAMI.partialPeriodicPattern.basic import GThreePGrowth  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'temporal_T10I4D100K.csv'
periodCount = 5000
periodicSupportCount = 100  #Users can also specify this constraint between 0 to 1.
relativePeriodSupp = 0.3

seperator='\t'       
```

#### Step 3: Execute the GThreePGrowth algorithm


```python
obj = alg.GThreePGrowth(iFile=inputFile,periodicSupport=periodicSupportCount, period=periodCount, relativePS=relativePeriodSupp, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Partial Periodic Patterns were generated successfully using G3PGrowth algorithm 


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='partialPeriodicPatternsMinSupCount100.txt')
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

    Total No of patterns: 9728


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 21.131616830825806


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 579915776
    Memory (USS): 568168448


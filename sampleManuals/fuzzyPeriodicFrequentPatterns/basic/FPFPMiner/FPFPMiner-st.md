# Discovering Fuzzy Periodic Frequent Pattern in Big Data Using FPFPMiner Algorithm

In this tutorial, we will discuss the first approach to find Fuzzy Periodic Frequent Pattern in big data using FPFPMiner algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover Fuzzy Periodic Frequent Pattern using a single minimum support value


***

## <a id='basicApproach'>Basic approach: Executing FPFPMiner on a single dataset at a particular minimum support value</a>

#### Step 1: Import the FPFPMiner algorithm


```python
from PAMI.fuzzyPeriodicFrequentPattern.basic import FPFPMiner  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'fuzzyTemporal_T10I4D100K.csv'
periodCount=1000
minimumSupportCount=1000  #Users can also specify this constraint between 0 to 1.

seperator=' '
```

#### Step 3: Execute the FPFPMiner algorithm


```python
obj = alg.FPFPMiner(iFile=inputFile, minSup=minimumSupportCount,period=periodCount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='frequentPatternsMinSupCount1000.txt')
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

    Total No of patterns: 382


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 336.29632902145386


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 427794432
    Memory (USS): 388661248


# Discovering Partial Periodic Pattern in Big Data Using PPP_ECLAT Algorithm

In this tutorial, we will discuss the first approach to find Partial Periodic Pattern in big data using PPP_ECLAT (3PECLAT) algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover Partial Periodic Pattern using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing PPP_ECLAT on a single dataset at a particular minimum support value</a>

#### Step 1: Import the PPP_ECLAT algorithm


```python
from PAMI.partialPeriodicPattern.basic import PPP_ECLAT  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'temporal_T10I4D100K.csv'
periodCount = 5000
periodicSupportCount = 100  #Users can also specify this constraint between 0 to 1.

seperator='\t'       
```

#### Step 3: Execute the PPP_ECLAT algorithm


```python
obj = alg.PPP_ECLAT(iFile=inputFile,periodicSupport=periodicSupportCount, period=periodCount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Partial Periodic Frequent patterns were generated successfully using 3PEclat algorithm


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='partialPeriodicPatternsPSCount100.txt')
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

    Runtime: 34.25913119316101


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 225923072
    Memory (USS): 187764736


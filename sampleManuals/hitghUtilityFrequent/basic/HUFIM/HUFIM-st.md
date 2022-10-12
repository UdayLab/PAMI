# Discovering High Utility Frequent patterns in Big Data Using HUFIM Algorithm

In this tutorial, we will discuss the first approach to find High Utility Frequent patterns in big data using HUFIM algorithm.

[__Basic approach:__](#basicApproach) Here, we generalize the basic approach by presenting the steps to discover High Utility Frequent patterns using single multiple minimum support value.

***

## <a id='basicApproach'>Basic approach: Executing HUFIM on a single dataset at a particular minimum support value</a>

#### Step 1: Import the HUFIM algorithm


```python
from PAMI.highUtilityFrequentPatterns.basic import HUFIM  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'utility_mushroom.txt'

minimumSupportCount = 2000  #Users can also specify this constraint between 0 to 1.
minUtilCount = 30000
seperator = ' '       
```

#### Step 3: Execute the HUFIM algorithm


```python
obj = alg.HUFIM(iFile=inputFile,minUtil=minUtilCount, minSup=minimumSupportCount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    High Utility Frequent patterns were generated successfully using HUFIM algorithm


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='utilityFrequentPatterns.txt')
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

    Total No of patterns: 6610


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 7.107502460479736


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 131117056
    Memory (USS): 92897280



```python

```

# Discovering Sequential Frequent Patterns in Big Data Using Spade Algorithm

In this tutorial, we will discuss two approaches to find frequent patterns in big data using Spade algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover frequent patterns using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing Spade on a single dataset at a particular minimum support value</a>

#### Step 1: Import the Spade algorithm


```python
from PAMI.sequentialPatternMining.basic import SPADE as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'sequence_retail.txt'

minimumSupportCount=1000  #Users can also specify this constraint between 0 to 1.

seperator='\t'       
```

#### Step 3: Execute the Spade algorithm


```python
obj = alg.SPADE(iFile=inputFile, minSup=minimumSupportCount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    1000
    Sequential Frequent patterns were generated successfully using SPADE algorithm 


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

    Total No of patterns: 181


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 6.3317389488220215


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 165269504
    Memory (USS): 126648320



```python

```

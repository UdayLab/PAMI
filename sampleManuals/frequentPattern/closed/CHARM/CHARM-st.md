# Discovering Closed Frequent patterns in Big Data Using CHARM Algorithm

In this tutorial, we will discuss the first approach to find Closed Frequent patterns in big data using CHARM algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover Closed Frequent patterns using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing CHARM on a single dataset at a particular minimum support value</a>

#### Step 1: Import the CHARM algorithm


```python
from PAMI.frequentPattern.closed import CHARM  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'transactional_T10I4D100K.csv'

minimumSupportCount=100  #Users can also specify this constraint between 0 to 1.

seperator='\t'       
```

#### Step 3: Execute the CHARM algorithm


```python
obj = alg.CHARM(iFile=inputFile, minSup=minimumSupportCount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Closed Frequent patterns were generated successfully using CHARM algorithm


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='closedPatternsMinSupCount100.txt')
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

    Total No of patterns: 17145


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 18.297893047332764



```python
##### Step 5.3: Total Memory consumed by the mining algorithm
```


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 139325440
    Memory (USS): 100913152


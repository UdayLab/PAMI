# Discovering Frequent Patterns in Uncertain Big Data Using TubeS Algorithm

In this tutorial, we will discuss two approaches to find frequent patterns in uncertain big data using TubeS algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover frequent patterns using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing TubeS on a single dataset at a particular minimum support value</a>

#### Step 1: Import the TubeS algorithm


```python
from PAMI.uncertainFrequentPattern.basic import TubeS as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'uncertainTransaction_T10I4D200K.csv'

minimumSupportCount = 300  #Users can also specify this constraint between 0 to 1.
seperator = '\t'       
```

#### Step 3: Execute the TubeS algorithm


```python
obj = alg.TubeS(iFile=inputFile, minSup=minimumSupportCount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Total number of false patterns generated: 1969
    Uncertain Frequent patterns were generated successfully using TubeS algorithm


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save('frequentPatternsMinSupCount400.txt')
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

    Total No of patterns: 874


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 297.2531898021698


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 1206177792
    Memory (USS): 1167814656


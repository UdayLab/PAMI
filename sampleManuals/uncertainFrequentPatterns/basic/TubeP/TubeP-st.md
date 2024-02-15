# Discovering Frequent Patterns in Uncertain Big Data Using TubeP Algorithm

In this tutorial, we will discuss two approaches to find frequent patterns in uncertain big data using TubeP algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover frequent patterns using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing TubeP on a single dataset at a particular minimum support value</a>

#### Step 1: Import the PUFGrowth algorithm


```python
from PAMI.uncertainFrequentPattern.basic import TubeP as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'uncertainTransaction_T10I4D200K.csv'

minimumSupportCount = 300  #Users can also specify this constraint between 0 to 1.
seperator = '\t'       
```

#### Step 3: Execute the PUFGrowth algorithm


```python
obj = alg.TubeP(iFile=inputFile, minSup=minimumSupportCount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Total number of false patterns generated: 1969
    Uncertain Frequent patterns were generated successfully using TubeP algorithm


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save('frequentPatternsMinSupCount300.txt')
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

    Runtime: 292.42088198661804


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 1026658304
    Memory (USS): 988524544


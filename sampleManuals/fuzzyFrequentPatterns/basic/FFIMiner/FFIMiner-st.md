# Discovering Frequent Patterns in Big Data Using FFIMiner Algorithm

In this tutorial, we will discuss the first approach to find frequent patterns in big data using FFIMiner algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover frequent patterns using a single minimum support value


***

## <a id='basicApproach'>Basic approach: Executing FFIMiner on a single dataset at a particular minimum support value</a>

#### Step 1: Import the FFIMiner algorithm

```python
from PAMI.fuzzyFrequentPatterns.basic import FFIMiner_old  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'fuzzyTransactional_T10I4D100K.csv'

minimumSupportCount=1000  #Users can also specify this constraint between 0 to 1.

seperator=' '       
```

#### Step 3: Execute the FFIMiner algorithm


```python
obj = alg.FFIMiner(iFile=inputFile, minSup=minimumSupportCount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='fuzzyFrequentPatternsMinSupCount1000.txt')
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

    Total No of patterns: 383


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 331.8520133495331


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 425664512
    Memory (USS): 386793472


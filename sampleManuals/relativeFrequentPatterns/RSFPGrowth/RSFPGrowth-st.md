# Discovering Frequent Patterns Using Other Measures in Big Data Using RSFPGrowth Algorithm

In this tutorial, we will discuss the first approach to find frequent patterns in big data using RSFPGrowth algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover frequent patterns using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing RSFPGrowth on a single dataset at a particular minimum support value</a>

#### Step 1: Import the RSFPGrowth algorithm


```python
from PAMI.relativePatterns.basic import RSFPGrowth  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'transactional_T10I4D100K.csv'

minimumSupportCount=100  #Users can also specify this constraint between 0 to 1.
minRatioEx=0.5
seperator='\t'       
```

#### Step 3: Execute the RSFPGrowth algorithm


```python
obj = alg.RSFPGrowth(inputFile, minimumSupportCount, minRatioEx, seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Relative support frequent patterns were generated successfully using RSFPGrowth algorithm


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='relativeFrequentPatternsMinSupCount100.txt')
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

    Total No of patterns: 4890


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 12.39819622039795


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 400482304
    Memory (USS): 362418176


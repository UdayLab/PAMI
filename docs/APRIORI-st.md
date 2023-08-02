# Discovering Frequent Patterns in Big Data Using Apriori Algorithm

In this tutorial, we will discuss first approach to find frequent patterns in big data using Apriori algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover frequent patterns using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing Apriori on a single dataset at a particular minimum support value</a>

#### Step 1: Import the Apriori algorithm


```python
from PAMI.frequentPattern.basic import Apriori  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = '/userData/likhitha/new/frequentPattern/transactional_T10I4D100K.csv'

minimumSupportCount=1000  #Users can also specify this constraint between 0 to 1.

seperator='\t'       
```

#### Step 3: Execute the Apriori algorithm


```python
obj = alg.Apriori(iFile=inputFile, minSup=minimumSupportCount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Frequent patterns were generated successfully using Apriori algorithm 


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

    Total No of patterns: 385


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 486.54657435417175


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 267091968
    Memory (USS): 228372480


# Basic Tutorial on Implementing Frequent-Pattern Growth Algorithm

In this tutorial, we will discuss the first approach to find frequent patterns in big data using FPGrowth algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover frequent patterns using a single minimum support value

***

#### Step 1: Import the FP-growth algorithm


```python
from PAMI.frequentPattern.basic import FPGrowth  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'transactional_T10I4D100K.csv' #specify the name of database

minimumSupportCount=100  #Users can also specify this constraint between 0 to 1.

seperator='\t'       #the character used to seperate the items in a transaction
```

#### Step 3: Execute the Apriori algorithm


```python
obj = alg.FPGrowth(iFile=inputFile, minSup=minimumSupportCount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Frequent patterns were generated successfully using frequentPatternGrowth algorithm


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='frequentPatternsMinSupCount100.txt') #Storing the patterns in a file
```

##### Step 4.2. Storing the generated patterns in a data frame


```python
frequentPatternsDF= obj.getPatternsAsDataFrame()   #Getting the patterns in to a data frame
```

#### Step 5: Getting the statistics

##### Step 5.1: Total number of discovered patterns 


```python
print('Total No of patterns: ' + str(len(frequentPatternsDF))) 
```

    Total No of patterns: 27532


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 10.297713994979858


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 516739072
    Memory (USS): 478552064


***

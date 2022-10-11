# Discovering Fuzzy Correlated Patterns in Big Data Using FCPGrowth Algorithm

In this tutorial, we will discuss the first approach to find Fuzzy Correlated Patterns in big data using FCPGrowth algorithm.

 [__Basic approach:__](#basicApproach) Here, we present the steps to discover Fuzzy Correlated Patterns using a single minimum support value


***

## <a id='basicApproach'>Basic approach: Executing FCPGrowth on a single dataset at a particular minimum support value</a>

#### Step 1: Import the FCPGrowth algorithm


```python
from PAMI.fuzzyCorrelatedPattern.basic import FCPGrowth  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'fuzzy_T10I4D100K.csv'

minimumSupportCount=1200  #Users can also specify this constraint between 0 to 1.
ratioExample=0.8
seperator='\t'       
```

#### Step 3: Execute the FCPGrowth algorithm


```python
obj = alg.FCPGrowth(iFile=inputFile, minSup=minimumSupportCount,ratio=ratioExample,sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Fuzzy Correlated Patterns Successfully generated using FCPGrowth algorithms


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

    Total No of patterns: 318


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 260.4294843673706


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 475705344
    Memory (USS): 436994048


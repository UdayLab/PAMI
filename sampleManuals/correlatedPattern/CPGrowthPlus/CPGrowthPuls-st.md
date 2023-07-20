# Discovering Correlated Frequent Patterns in Big Data Using CPGrowthPlus Algorithm

In this tutorial, we will discuss the first approach to find correlated patterns in big data using CPGrowthPlus algorithm.
 [__Basic approach:__](#basicApproach) Here, we present the steps to discover correlated patterns using a single minimum support value


***

## <a id='basicApproach'>Basic approach: Executing CPGrowthPlus on a single dataset at a particular minimum support value</a>

#### Step 1: Import the CPGrowthPlus algorithm

```python
from PAMI.correlatedPattern.basic import CPGrowthPlus as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'transactional_T10I4D100K.csv'
minAllConfCount=0.1
minimumSupportCount=100  #Users can also specify this constraint between 0 to 1.

seperator='\t'       
```

#### Step 3: Execute the CPGrowthPlus algorithm


```python
obj = alg.CPGrowthPlus(iFile=inputFile, minSup=minimumSupportCount,  minAllConf=minAllConfCount ,sep=seperator)   #initialize
obj.startMine()            #Start the mining process
```

    Correlated Frequent patterns were generated successfully using CorrelatedPatternGrowth algorithm


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='correlatedPatternsMinSupCount100.txt')
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

    Total No of patterns: 5758


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 13.232695817947388



```python
##### Step 5.3: Total Memory consumed by the mining algorithm
```


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 401813504
    Memory (USS): 363274240


# Discovering Sequential Frequent Patterns in Big Data Using PrefixSpan Algorithm

In this tutorial, we will discuss two approaches to find frequent patterns in big data using PrefixSpan algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover frequent patterns using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing PrefixSpan on a single dataset at a particular minimum support value</a>

#### Step 1: Import the PrefixSpan algorithm


```python
from PAMI.sequentialPatternMining.basic import prefixSpan as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'sequence_retail.txt'

minimumSupportCount=1000  #Users can also specify this constraint between 0 to 1.

seperator='\t'       
```

#### Step 3: Execute the PrefixSpan algorithm


```python
obj = alg.prefixSpan(iFile=inputFile, minSup=minimumSupportCount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Frequent patterns were generated successfully using prefixSpan algorithm 


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

    Runtime: 6.181333303451538



```python
##### Step 5.3: Total Memory consumed by the mining algorithm
```


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 151408640
    Memory (USS): 112873472



```python

```

# Discovering Periodic Frequent Patterns in Uncertain Big Data Using UPFPGrowth++ Algorithm

In this tutorial, we will discuss two approaches to find periodic frequent patterns in uncertain big data using UPFPGrowth++ algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover periodic frequent patterns using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing UPFPGrowth++ on a single dataset at a particular minimum support value</a>

#### Step 1: Import the UPFPGrowth++ algorithm


```python
from PAMI.uncertainPeriodicFrequentPattern.basic import UPFPGrowthPlus as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'uncertainTemporal_T10I4D100K.csv'

minimumSupportCount = 300  #Users can also specify this constraint between 0 to 1.
maximumPeriodCount = 5000
seperator = '\t'       
```

#### Step 3: Execute the UPFPGrowth++ algorithm


```python
obj = alg.UPFPGrowthPlus(iFile=inputFile, minSup=minimumSupportCount, maxPer=maximumPeriodCount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Periodic Frequent patterns were generated successfully using Periodic-TubeP algorithm


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save('periodicFrequentPatternsMinSupCount300.txt')
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

    Total No of patterns: 480


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 15.995655536651611


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 707694592
    Memory (USS): 668655616



```python

```

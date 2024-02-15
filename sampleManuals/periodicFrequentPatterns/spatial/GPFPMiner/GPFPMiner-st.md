# Discovering Geo-Referenced (Spatial) Periodic Frequent patterns in Big Data Using PFS_ECLAT Algorithm

In this tutorial, we will discuss the first approach to find Geo-Referenced Periodic Frequent patterns in big data using GPFPMiner algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover Geo-Referenced Periodic Frequent patterns using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing GPFPMiner on a single dataset at a particular minimum support value</a>

#### Step 1: Import the GPFPMiner algorithm


```python
from PAMI.geoReferencedPeriodicFrequentPattern.basic import GPFPMiner  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'temporal_T10I4D100K.csv'
neighborFile = 't10_neighbours.txt'
minimumSupportCount = 100  #Users can also specify this constraint between 0 to 1.
maxmunPeriodCount = 5000
seperator = '\t'       
```

#### Step 3: Execute the GPFPMiner algorithm


```python
obj = alg.GPFPMiner(iFile=inputFile, minSup=minimumSupportCount,maxPer=maxmunPeriodCount, nFile=neighborFile,sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    100 5000
    Spatial Periodic Frequent patterns were generated successfully using SpatialEclat algorithm


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='periodicFrequentPatternsMinSupCount100.txt')
```

##### Step 4.2. Storing the generated patterns in a data frame


```python
periodicFrequentPatternsDF= obj.getPatternsAsDataFrame()
```

#### Step 5: Getting the statistics

##### Step 5.1: Total number of discovered patterns 


```python
print('Total No of patterns: ' + str(len(periodicFrequentPatternsDF)))
```

    Total No of patterns: 789


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 1.5601165294647217


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 222597120
    Memory (USS): 184107008


# Discovering Geo-Referenced (Spatial) Partial Periodic patterns in Big Data Using STECLAT Algorithm

In this tutorial, we will discuss the first approach to find Geo-Referenced Partial Periodic patterns in big data using STECLAT algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover Geo-Referenced Partial Periodic patterns using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing STECLAT on a single dataset at a particular minimum periodic support value</a>

#### Step 1: Import the STECLAT algorithm


```python
from PAMI.partialPeriodicSpatialPattern.basic import STEclat  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'temporal_T10I4D100K.csv'
neighborFile = 't10_neighbours.txt'
minimumSupportCount = 100  #Users can also specify this constraint between 0 to 1.
maxmunPeriodCount = 40000
seperator = '\t'       
```

#### Step 3: Execute the STECLAT algorithm


```python
obj = alg.STEclat(iFile=inputFile, minPS=minimumSupportCount,maxIAT=maxmunPeriodCount, nFile=neighborFile,sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Spatial Periodic Frequent patterns were generated successfully using SpatialEclat algorithm


#### Step 4: Storing the generated patterns

##### Step 4.1: Storing the generated patterns in a file


```python
obj.save(outFile='spatialPeriodicPatternsMinSupCount100.txt')
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

    Total No of patterns: 797


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 1.404829740524292


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 222351360
    Memory (USS): 183980032


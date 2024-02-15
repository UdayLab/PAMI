# Discovering Frequent Patterns in Big Data Using parallelECLAT Algorithm

In this tutorial, we will discuss the first approach to find frequent patterns in big data using parallelECLAT algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover frequent patterns using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing parallelECLAT on a single dataset at a particular minimum support value</a>

#### Step 1: Import the parallelECLAT algorithm


```python
from PAMI.frequentPattern.pyspark import parallelECLAT  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'transactional_T10I4D100K.csv'

minimumSupportCount=100  #Users can also specify this constraint between 0 to 1.
mumberWorkersCount=4
seperator='\t'       
```

#### Step 3: Execute the parallelECLAT algorithm


```python
obj = alg.parallelECLAT(iFile=inputFile, minSup=minimumSupportCount,numWorkers=mumberWorkersCount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
    22/10/10 09:43:22 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
                                                                                    

    Frequent patterns were generated successfully using Parallel ECLAT algorithm


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

    Total No of patterns: 27532


##### Step 5.2: Runtime consumed by the mining algorithm


```python
print('Runtime: ' + str(obj.getRuntime()))
```

    Runtime: 14.83729887008667


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 225456128
    Memory (USS): 186662912


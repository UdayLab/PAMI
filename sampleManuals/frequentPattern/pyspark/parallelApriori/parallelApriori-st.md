# Discovering Frequent Patterns in Big Data Using parallelApriori Algorithm

In this tutorial, we will discuss the first approach to find frequent patterns in big data using parallelApriori algorithm.

[__Basic approach:__](#basicApproach) Here, we present the steps to discover frequent patterns using a single minimum support value

***

## <a id='basicApproach'>Basic approach: Executing parallelApriori on a single dataset at a particular minimum support value</a>

#### Step 1: Import the parallelApriori algorithm


```python
from PAMI.frequentPattern.pyspark import parallelApriori  as alg
```

#### Step 2: Specify the following input parameters


```python
inputFile = 'transactional_T10I4D100K.csv'

minimumSupportCount=1000  #Users can also specify this constraint between 0 to 1.
mumberWorkersCount=4
seperator='\t'       
```

#### Step 3: Execute the parallelApriori algorithm


```python
obj = alg.parallelApriori(iFile=inputFile, minSup=minimumSupportCount,numWorkers=mumberWorkersCount, sep=seperator)    #initialize
obj.startMine()            #Start the mining process
```

    Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
    22/10/10 09:50:35 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
    22/10/10 09:50:36 WARN Utils: Service 'SparkUI' could not bind on port 4040. Attempting port 4041.
                                                                                    

    Frequent patterns were generated successfully using Parallel Apriori algorithm


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

    Runtime: 263.0825595855713


##### Step 5.3: Total Memory consumed by the mining algorithm


```python
print('Memory (RSS): ' + str(obj.getMemoryRSS()))
print('Memory (USS): ' + str(obj.getMemoryUSS()))
```

    Memory (RSS): 129470464
    Memory (USS): 91033600


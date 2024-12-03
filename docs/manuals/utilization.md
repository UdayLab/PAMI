[Previous](terminalExecute.html)|[🏠 Home](index.html)|[Next](evaluateMultipleAlgorithms.html)

## Mining interesting patterns using an algorithm

In this page, we first describe and illustrate the basic process to implement a mining algorithm and store the results.
Next, we describe the process to evaluate a mining algorithm at different constraint values in a database.

## 1. Implementing a pattern mining algorithm on a dataset at a single constraint value

### Syntax
```Python
# import the necessary algorithm from the PAMI library
from PAMI.<model>.<basic/closed/maximal/topK> import <algorithmName> as alg

# Call the necessary algorithm by passing necessary input parameters. The input parameters include inputFileName and the user-specified constraints.
obj = alg.<algorithmName>(<input parameters>)

# Start the mining algorithm
obj.mine()

# Collect the patterns discovered by the algorithm in the database
discoveredPatterns = obj.getDiscoveredPatterns()

# Print the total number of patterns
print("Total number of discovered patterns:", len(discoveredPatterns))

# Store the discovered patterns in a file. 
obj.save('<outputFileName>')

# Output the discovered patterns as a data frame
Df = obj.getPatternInDataFrame()

# Calculate the [USS] memory consumed by the algorithm
print("Total Memory in USS:", obj.getMemoryUSS())

# Calculate the RSS memory consumed by the algorithm. We suggest using RSS memory for the memory comparison
print("Total Memory in RSS", obj.getMemoryRSS())

# Calculate the runtime requirements by the algorithm
print("Total ExecutionTime in seconds:", obj.getRuntime())
```


### Example 1: Mining frequent patterns in a transactional database using FP-growth

__Note:__ [Click here to download the transactional database](https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/Transactional_T10I4D100K.csv)

```Python
#import the frequent pattern mining algorithm
from PAMI.frequentPattern.basic import FPGrowth as alg

#inputFile = 'fileName'
inputFile = 'Transactional_T10I4D100K.csv'

#specify the constraints used in the model
minSup=400

#create the object of the mining algorithm 
obj = alg.FPGrowth(inputFile, minSup)

#start the mining process
obj.mine()

#Print the number of interesting patterns generated
print("Total number of Frequent Patterns:", len(obj.getPatterns()))

#Save the generated patterns in a file
obj.save('outputFile.tsv')

# Determine the memory consumed by the mining algorithm
print("Total Memory in RSS", obj.getMemoryRSS())

# Determine the total runtime consumed by the mining algorithm
print("Total ExecutionTime in seconds:", obj.getRuntime())
```

## 2. Implementing a pattern mining algorithm on a dataset at multiple constraint values

### Syntax

```Python
# import the necessary algorithm from the PAMI library
from PAMI. < model >.< basic / closed / maximal / topK >
import < algorithmName > as alg
algorithm = 'Algorithm Name'

# Create a data frame to store the values
import pandas as pd

result = pd.DataFrame(columns=['algorithm', 'minSup', 'patterns', 'runtime', 'memory'])

# Specify the List of constraint values
constraintList = ['List of values']

# For each value in the constraint list
for constraint in constraintList:
    # Call the necessary algorithm by passing necessary input parameters. The input parameters include inputFileName and the user-specified constraints.
    obj = alg. < algorithmName > (< input parameters >)

    # Start the mining algorithm
    obj.mine()

    # store the results in the data frame
    result.loc[result.shape[0]] = [algorithm, constraint, len(obj.getPatterns()), obj.getRuntime(), obj.getMemoryRSS()]

# Print the dataframe
result

# Visualize the plots
from PAMI.extras.graph import DF2Fig as dif
# Pass the result data frame to the class
ab = dif.dataFrameInToFigures(result)
# Draw the graphs
ab.plotGraphsFromDataFrame()
```

### Example 2: Mining frequent patterns in a transactional database at different mininimum support values using FP-growth

```Python
#Import the mining algorithm
from PAMI.frequentPattern.basic import FPGrowth as alg

#Import pandas data frame to store the values 
import pandas as pd

#Initialize the data frame
result = pd.DataFrame(columns=['algorithm', 'minSup', 'patterns', 'runtime', 'memory'])

#Specify the name of the input file
inputFile = 'Transactional_T10I4D100K.csv'

#specify the seperator
seperator = '\t'

#Create a list of constraint values
minimumSupportList = [500, 600, 700, 800]
#minimumSupport values can be specified between 0 to 1.
#Example: minSupList = [0.005, 0.006, 0.007, 0.008, 0.009]

#specify the name of the algorithm
algorithm = 'FP-growth'  #specify the algorithm name

#Run the for loop for each minSup value
for minSup in minimumSupportList:
    #Create the object
    obj = alg.FPGrowth(inputFile, minSup, seperator)
    #start the mining process
    obj.mine()
    #store the results in the data frame
    result.loc[result.shape[0]] = [algorithm, minSup, len(obj.getPatterns()), obj.getRuntime(), obj.getMemoryRSS()]

#Print the dataframe
result

#Visualize the plots
from PAMI.extras.graph import DF2Fig as dif

#Pass the result data frame to the class
ab = dif.dataFrameInToFigures(result)
#Draw the graphs
ab.plotGraphsFromDataFrame()
```

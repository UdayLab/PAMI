**[CLICK HERE](manual.html)** to access the PAMI manual.


# Methods to utilize PAMI library

The PAMI library can be utilized either as a library in any Python program or a command line based stand-alone application. We now describe both of these methods.

**Pre-requisites:** 


Please ensure that PAMI library was already installed on your machine. The manual for installing PAMI can be found [here](installation.html).

## Utilizing PAMI library in a Python program
1. If the PAMI library was installed using 'pip' command, then skip the next step and move to step 3.
1. If the PAMI library was cloned/downloaded from the GitHub, then copy the PAMI source package into your project folder. 
    **The sub-folder with the name of PAMI represents the source package.**
1. The syntax to utilize an algorithm in PAMI is as follows:

```Python

# import the necessary algorithm from the PAMI library
from PAMI.<model>.<basic/closed/maximal/topK> import <algorithmName> as alg

# Call the necessary algorithm by passing necessary input parameters. The input parameters include inputFileName and the user-specified constraints.
obj = alg.<algorithmName>(<input parameters>)

# Start the mining algorithm
obj.startMine()

# Collect the patterns discovered by the algorithm in the database
discoveredPatterns = obj.getDiscoveredPatterns()

# Print the total number of patterns
print("Total number of discovered patterns:", len(discoveredPatterns))

# Store the discovered patterns in a file. 
obj.storePatternsInFile('<outputFileName>')

# Output the discovered patterns as a data frame
Df = obj.getPatternInDataFrame()

# Calculate the [USS] memory consumed by the algorithm
print("Total Memory in USS:", obj.getMemoryUSS())

# Calculate the RSS memory consumed by the algorithm. We suggest using RSS memory for the memory comparison
print("Total Memory in RSS", obj.getMemoryRSS())

# Calculate the runtime requirements by the algorithm
print("Total ExecutionTime in seconds:", obj.getRuntime())


```
### Example: Using FP-growth algorithm to find frequent patterns in a transactional database.

```Python

from PAMI.frequentPattern.basic import fpGrowth as alg

obj = alg.fpGrowth('inputTransactionalDatabase.tsv', minSup)
obj.startMine()
frequentPatterns = obj.getDiscoveredPatterns()
print("Total number of Frequent Patterns:", len(frequentPatterns))
obj.storePatternsInFile('outputFile.tsv')
print("Total Memory in RSS", obj.getMemoryRSS())
print("Total ExecutionTime in seconds:", obj.getRuntime())

```
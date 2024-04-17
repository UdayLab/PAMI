[__<--__ Return to home page](index.html)

## Inputs and ouputs of an algorithm in PAMI

## Inputs
When you are writing algorithms for PAMI, please ensure the following:

1. Dataset
   - text file or
   - data frame
   
       The data frame will contain multiple rows but only single column. In text file or data frame, items are seperated by a seperator.
        
2. Algorithm specific constraints. For example, minimum support, maximum periodicity, and top-k.

## Outputs
The algorithm developers were requested to ensure the following output methods exist:

1. __save():__ function must exist to save the generated patterns in a file.
2. __getPatternsAsDataFrame():__ function that outputs the generated patterns into a data frame
3. __getMemoryUSS():__ function must output the USS memory consumed by a program.
4. __getMemoryRSS():__ function must output the RSS memory consumed by a program.
5. __getRuntime():__ function must output the runtime consumed by a program.
6. __printResults():__ function must print the results of a program. This function is provided below.

```Python
def printResults(self):
    print("Total number of Frequent Patterns:", len(self.getPatterns()))
    print("Total Memory in USS:", self.getMemoryUSS())
    print("Total Memory in RSS", self.getMemoryRSS())
    print("Total ExecutionTime in ms:", self.getRuntime())
```

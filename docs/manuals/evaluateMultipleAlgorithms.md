[__<--__ Previous ](utilization.html)|[Home](evaluateMultipleAlgorithms.html)|[_Next_-->](visualizeSpatialItems.html)


## Evaluating Multiple Pattern Mining Algorithms on a Dataset

This page we provide the step-by-step process to evaluate multiple pattern mining algorithms at different input parameter values on a dataset. We use frequent pattern mining algorithms, Apriori, FP-growth, and ECLAT, for illustration purposes. 

#### Step 1: Creation of pandas data frame to store the results of multiple algorithms

```Python
import pandas as pd
result = pd.DataFrame(columns=['algorithm', 'minSup', 'patterns', 'runtime', 'memory'])
```
__Note:__ The first column of data frame has to be 'algorithm.'  Otherwise, the code will generate the error.

#### Step 2: Declaring the name of the dataset

```Python
#inputFile = 'fileName'
inputFile = 'Transactional_T10I4D100K.csv'
```

[Click here to download the dataset.](https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/Transactional_T10I4D100K.csv)

#### Step 3: Specify the range of values for an input parameter

```Python
#constraintList = [array of values]  
minSupList = [400,500,600,700,800,900,1000]
```

#### Step 4: Declare the algorithm name, import and execute it, and store the results in the data frame

```Python
#algorithmName = 'name of the algorithm'
algorithmName = 'Apriori'

#import the mining algorithm
from PAMI.frequentPattern.basic import Apriori as alg

# execute the mining algorithm at different constraint values using the for loop
#for constraint in constraintList:
for minSup in minSupList:
    #create an object of the mining algorithm 
    obj = alg.Apriori(inputFile,minSup, sep='\t')

    #start the mining process
    obj.startMine()

    #append the results into the data frame
    result.loc[result.shape[0]] = [algorithmName, minSup, len(obj.getPatterns()), obj.getRuntime(), obj.getMemoryRSS()]
```

#### Step 5: Repeat Step-4 for each other pattern mining algorithms

```Python
#algorithmName = 'name of the algorithm'
algorithmName = 'FPGrowth'

#import the mining algorithm
from PAMI.frequentPattern.basic import FPGrowth as alg

# execute the mining algorithm at different constraint values using the for loop
#for constraint in constraintList:
for minSup in minSupList:
    #create an object of the mining algorithm 
    obj = alg.FPGrowth(inputFile,minSup, sep='\t')

    #start the mining process
    obj.startMine()

    #append the results into the data frame
    result.loc[result.shape[0]] = [algorithmName, minSup, len(obj.getPatterns()), obj.getRuntime(), obj.getMemoryRSS()]

#---------------------------------
#Repeating above steps for another algorithm

#algorithmName = 'name of the algorithm'
algorithmName = 'ECLAT'

#import the mining algorithm
from PAMI.frequentPattern.basic import ECLAT as alg

# execute the mining algorithm at different constraint values using the for loop
#for constraint in constraintList:
for minSup in minSupList:
    #create an object of the mining algorithm 
    obj = alg.ECLAT(inputFile,minSup, sep='\t')

    #start the mining process
    obj.startMine()

    #append the results into the data frame
    result.loc[result.shape[0]] = [algorithmName, minSup, len(obj.getPatterns()), obj.getRuntime(), obj.getMemoryRSS()]
```
#### Step 6: Visualizing the comparative results

```Python
# Import the library 
from PAMI.extras.graph import DF2Fig as dif
# Pass the result data frame to the class
ab = dif.dataFrameInToFigures(result)
# Draw the graphs
ab.plotGraphsFromDataFrame()
```

#### Step 7: Creating Latex files for graphs

```Python
# Import the library
from PAMI.extras.graph import DF2Tex as gdf
# Pass the result data frame
gdf.generateLatexCode(result)
```
__Note:__ The _generateLatexCode_ program create three latex files, namely _patternsLatexfile.tex_, _memoryLatexfile.tex_, and _runtimeLatexfile.tex_. 

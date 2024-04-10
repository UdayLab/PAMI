[__<--__ Previous ](DenseFormatDF.html)|[Home](index.html)|[_Next_-->](createTransactionalDatabase.html)

# Sparse dataframe

### Introduction
A sparse dataframe is basically a (non-sparse) matrix in which the first column represents the row-identifier/timestamp, 
the second column represents the item, and the third column represents the value of the corresponding item.
The format of the sparse dataframe is as follows:

      rowIdentifier/timestamp   Item1   Value

An example of a dense dataframe generated from the customer purchase database is as follows:

  timestamp | Item | Value
  ---------|-----|---
    1| Bread | 3
    1|Jam|1
    1|Butter|2
    2|Bread|7
    2|Jam|2
   ...|...|...

### Converting a sparse dataframe into different database formats
Currently, PAMI supports converting a dataframe into a transactional database, temporal database, ond a utility database.
The users can avail this support by employing the methods available in **dataPreprocessign.SparseFormatDF** class.  
We now present these three methods.

#### Converting a dense dataframe into a transactional database
A [transactional database](transactionalDatabase.html) represents a sparse and binary representation of items occurring in a dataframe. 
The steps to convert a dataframe into a transactional database is as follows:

1. Initialize the SparseFormatDF class by passing the following three parameters: 
   1. inputDataFrame  - the dataframe that needs to converted into a database
   1. thresholdValue  - this value will be used to convert a non-binary data frame into a binary database
   1. condition       - The condition that needs to employed on the threshold value. Currently, the users can specify 
      the following six constraints: >, >=, <, <=, ==, and !=.

1. Call 'createTransactional(outputFileName)' method to store the dataframe as a transactional database.

A sample program to convert a dataframe into a transactional database and use it in a pattern mining algorithm, say FP-growth, is provided below

 ```Python
from PAMI.extras.DF2DB import SparseFormatDF as pro
from PAMI.frequentPattern.basic import FPGrowth as alg
import pandas as pd

# Objective: convert the above dataframe into a transactional database with items whose value is greater than or equal 1.
db = pro.SparseFormatDF(inputDataFrame=pd.DataFrame('mentionDataFrame'), thresholdValue=1, condition='>=')
# Convert and store the dataframe as a transactional database file
db.createTransactional(outputFile='/home/userName/transactionalDB.txt')
# Getting the fileName of the transactional database
print('The output file is saved at ' + db.getFileName())

# Using the generated transactional database in FP-growth algorithm to discover frequent patterns

obj = alg.fpGrowth(iFile=db.getFileName(), minSup='10.0')
obj.startMine()
patternsDF = obj.getPatternsAsDataFrame()

   ```

#### Converting a dense dataframe into a temporal database
A [temporal database](temporalDatabase.html) represents a sparse and binary representation of items occurring at a particular timestamp
in a dataframe.  The steps to convert a dataframe into a temporal database is as follows:

1. Initialize the SparseFormatDF class by passing the following three parameters: 
   1. inputDataFrame  - the dataframe that needs to converted into a database
   1. thresholdValue  - this value will be used to convert a non-binary data frame into a binary database
   1. condition       - The condition that needs to employed on the threshold value. Currently, the users can specify 
      the following six constraints: >, >=, <, <=, ==, and !=.

1. Call 'createTemporal(outputFileName)' method to store the dataframe as a temporal database.

A sample program to convert a dataframe into a temporal database and use it in a pattern mining algorithm, say PFP-growth++, is provided below

 ```Python
   from PAMI.extras.DF2DB import SparseFormatDF as pro
from PAMI.periodicFrequentPattern.basic import PFPGrowthPlus as alg
import pandas as pd

# Objective: convert the above dataframe into a transactional database with items whose value is greater than or equal 1.
db = pro.SparseFormatDF(inputDataFrame=pd.DataFrame('mentionDataFrame'), thresholdValue=1, condition='>=')
# Convert and store the dataframe as a transactional database file
db.createTransactional(outputFile='/home/userName/temporalDB.txt')
# Getting the fileName of the transactional database
print('The output file is saved at ' + db.getFileName())

obj = alg.PFPGrowthPlus(db.getFileName(), minSup="2", maxPer="6")
obj.startMine()
patternsDF = obj.getPatternsAsDataFrame()

``` 
#### Converting a dense dataframe into a utility database
A [utility database](utilityDatabase.html) represents a sparse and non-binary representation of items occurring in
each row of a dataframe.  The steps to convert a dataframe into a utility database is as follows:

1. Initialize the SparseFormatDF class by passing the following three parameters: 
   1. inputDataFrame  - the dataframe that needs to converted into a database
   1. thresholdValue  - this value will be used to convert a non-binary data frame into a binary database
   1. condition       - The condition that needs to employed on the threshold value. Currently, the users can specify 
      the following six constraints: >, >=, <, <=, ==, and !=.

1. Call 'createUtility(outputFileName)' method to store the dataframe as a temporal database.

A sample program to convert a dataframe into a utility database and use it in a pattern mining algorithm, say EFIM, is provided below

 ```Python
   from PAMI.extras.DF2DB import SparseFormatDF as pro
from PAMI.highUtilityPattern.basic import EFIM as alg
import pandas as pd

# Objective: convert the above dataframe into a transactional database with items whose value is greater than or equal 1.
db = pro.SparseFormatDF(inputDataFrame=pd.DataFrame('mentionDataFrame'), thresholdValue=1, condition='>=')
# Convert and store the dataframe as a transactional database file
db.createTransactional(outputFile='/home/userName/utilityDB.txt')
# Getting the fileName of the transactional database
print('The output file is saved at ' + db.getFileName())

  ```

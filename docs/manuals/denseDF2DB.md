[Return to home page](index.html)  

## Dense dataframe

### Description
A dense dataframe is basically a  matrix in which the first column represents the row-identifier/timestamp
and the remaining columns represent the items and their values. T

### Format of a dense dataframe 

      rowIdentifier/timestamp   Item1   Item2   ... ItemN

### An example 

  timestamp | Bread | Jam | Butter | Books | Pencil
  ---------|-----|---|------|---|------
    1| 3 | 1| 2|0 |0
    2|7|2|0|10|20
    3|0|0|3|0|0
    4|4|0|0|0|0

In the above dataframe (or table), the first transaction (or row) provides the information that a customer has purchased the 3 packets 
of Bread, 1 bottle of Jam, 3 packets of Butter at the timestamp of 1. The second transaction provides the information
that a customer has purchased 7 packets of Bread, 2 bottles of Jam, 10 Books and 20 Pencils. Similar arguments can be 
made for the remaining transactions in the above dataframe.

## Conversion of dense dataframe into different database formats
Currently, PAMI supports converting a dataframe into a transactional database, temporal database, ond a utility database.
The users can avail this support by employing the functions available in **dataPreprocessing.dense2DB** class.  

### Dense dataframe to transactional database
#### Description
A [transactional database](transactionalDatabase.html) represents a sparse and binary representation of items occurring in a dataframe. 
The steps to convert a dataframe into a transactional database is as follows:

1. Initialize the dense2DB class by passing the following three parameters: 
   1. inputDataFrame  - the dataframe that needs to converted into a database
   1. thresholdValue  - this value will be used to convert a non-binary data frame into a binary database
   1. condition       - The condition that needs to employed on the threshold value. Currently, the users can specify 
      the following six constraints: >, >=, <, <=, ==, and !=.

1. Call 'createTransactional(outputFileName)' method to store the dataframe as a transactional database.

#### Sample code

 ```Python
from PAMI.extras.DF2DB import denseDF2DB as pro 
import pandas as pd

# Objective: convert the above dataframe into a transactional database with items whose value is greater than or equal 1.
db = pro.dense2DB(inputDataFrame=pd.DataFrame('mentionDataFrame'), thresholdValue=1, condition='>=')
# Convert and store the dataframe as a transactional database file
db.createTransactionalDatabase(outputFile='/home/userName/transactionalDB.txt')
# Getting the fileName of the transactional database
print('The output file is saved at ' + db.getFileName())

   ```

### Dense dataframe to a temporal database
#### Description
A [temporal database](temporalDatabase.html) represents a sparse and binary representation of items occurring at a particular timestamp
in a dataframe.  The steps to convert a dataframe into a temporal database is as follows:

1. Initialize the dense2DB class by passing the following three parameters: 
   1. inputDataFrame  - the dataframe that needs to converted into a database
   1. thresholdValue  - this value will be used to convert a non-binary data frame into a binary database
   1. condition       - The condition that needs to employed on the threshold value. Currently, the users can specify 
      the following six constraints: >, >=, <, <=, ==, and !=.

1. Call 'createTemporal(outputFileName)' method to store the dataframe as a temporal database.

#### Sample code 
 ```Python
from PAMI.extras.DF2DB import denseDF2DB as pro 
import pandas as pd

# Objective: convert the above dataframe into a transactional database with items whose value is greater than or equal 1.
db = pro.dense2DB(inputDataFrame=pd.DataFrame('mentionDataFrame'), thresholdValue=1, condition='>=')
# Convert and store the dataframe as a temporal database file
db.createTemporalDatabase(outputFile='/home/userName/temporalDB.txt')
# Getting the fileName of the temporal database
print('The output file is saved at ' + db.getFileName())

``` 
### Converting a dense dataframe into a utility database
#### Description
A [utility database](utilityDatabase.html) represents a sparse and non-binary representation of items occurring in
each row of a dataframe.  The steps to convert a dataframe into a utility database is as follows:

1. Initialize the dense2DB class by passing the following three parameters: 
   1. inputDataFrame  - the dataframe that needs to converted into a database
   1. thresholdValue  - this value will be used to convert a non-binary data frame into a binary database
   1. condition       - The condition that needs to employed on the threshold value. Currently, the users can specify 
      the following six constraints: >, >=, <, <=, ==, and !=.

1. Call 'createUtility(outputFileName)' method to store the dataframe as a temporal database.

#### Sample code
 ```Python
from PAMI.extras.DF2DB import denseDF2DB as pro 
import pandas as pd

# Objective: convert the above dataframe into a utility database with items whose value is greater than or equal 1.
db = pro.dense2DB(inputDataFrame=pd.DataFrame('mentionDataFrame'), thresholdValue=1, condition='>=')
# Convert and store the dataframe as a utility database file
db.createUtilityDatabase(outputFile='/home/userName/utilityDB.txt')
# Getting the fileName of the utility database
print('The output file is saved at ' + db.getFileName())

  ```

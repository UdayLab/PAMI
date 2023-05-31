[Return to home page](index.html)  

## Converting a sparse dataframe into different database formats
Currently, PAMI supports converting a dataframe into a transactional database, temporal database, ond a utility database.
The users can avail this support by employing the methods available in **dataPreprocessign.sparseDF2DB** class.  
We now present these three methods.

### Sparse dataframe into a transactional database
#### Description
A [transactional database](transactionalDatabase.html) represents a sparse and binary representation of items occurring in a dataframe. 
The steps to convert a dataframe into a transactional database is as follows:

1. Initialize the sparseDF2DB class by passing the following three parameters: 
   1. inputDataFrame  - the dataframe that needs to converted into a database
   1. thresholdValue  - this value will be used to convert a non-binary data frame into a binary database
   1. condition       - The condition that needs to employed on the threshold value. Currently, the users can specify 
      the following six constraints: >, >=, <, <=, ==, and !=.

1. Call 'createTransactional(outputFileName)' method to store the dataframe as a transactional database.

#### Sample code

 ```Python
from PAMI.extras.DF2DB import sparseDF2DB as pro 
import pandas as pd

# Objective: convert the above dataframe into a transactional database with items whose value is greater than or equal 1.
db = pro.sparseDF2DB(inputDF=pd.read_csv('mentionDataFrame'), thresholdValue=1, condition='>=')
# Convert and store the dataframe as a transactional database file
db.createTransactionalDatabase(outputFile='/home/userName/transactionalDB.txt')
# Getting the fileName of the transactional database
print('The output file is saved at ' + db.getFileName())


   ```

### Sparse dataframe into a temporal database
#### Description
A [temporal database](temporalDatabase.html) represents a sparse and binary representation of items occurring at a particular timestamp
in a dataframe.  The steps to convert a dataframe into a temporal database is as follows:

1. Initialize the sparseDF2DB class by passing the following three parameters: 
   1. inputDataFrame  - the dataframe that needs to converted into a database
   1. thresholdValue  - this value will be used to convert a non-binary data frame into a binary database
   1. condition       - The condition that needs to employed on the threshold value. Currently, the users can specify 
      the following six constraints: >, >=, <, <=, ==, and !=.

1. Call 'createTemporal(outputFileName)' method to store the dataframe as a temporal database.

#### Sample code

 ```Python
from PAMI.extras.DF2DB import sparseDF2DB as pro
from PAMI.periodicFrequentPattern.basic import PFPGrowthPlus as alg
import pandas as pd

# Objective: convert the above dataframe into a transactional database with items whose value is greater than or equal 1.
db = pro.sparseDF2DB(inputDataFrame=pd.read_csv('mentionDataFrame'), thresholdValue=1, condition='>=')
# Convert and store the dataframe as a temporal database file
db.createTemporalDatabase(outputFile='/home/userName/temporalDB.txt')
# Getting the fileName of the temporal database
print('The output file is saved at ' + db.getFileName())

``` -
### Sparse dataframe into a utility database
#### Description
A [utility database](utilityDatabase.html) represents a sparse and non-binary representation of items occurring in
each row of a dataframe.  The steps to convert a dataframe into a utility database is as follows:

1. Initialize the sparseDF2DB class by passing the following three parameters: 
   1. inputDataFrame  - the dataframe that needs to converted into a database
   1. thresholdValue  - this value will be used to convert a non-binary data frame into a binary database
   1. condition       - The condition that needs to employed on the threshold value. Currently, the users can specify 
      the following six constraints: >, >=, <, <=, ==, and !=.

1. Call 'createUtility(outputFileName)' method to store the dataframe as a temporal database.

#### Sample code

 ```Python
from PAMI.extras.DF2DB import sparseDF2DB as pro
from PAMI.highUtilityPatterns.basic import EFIM as alg
import pandas as pd

# Objective: convert the above dataframe into a utility database with items whose value is greater than or equal 1.
db = pro.sparseDF2DB(inputDataFrame=pd.read_csv('mentionDataFrame'), thresholdValue=1, condition='>=')
# Convert and store the dataframe as a utility database file
db.createUtilityDatabase(outputFile='/home/userName/utilityDB.txt')
# Getting the fileName of the utility database
print('The output file is saved at ' + db.getFileName())

  ```

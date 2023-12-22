[__<--__ Return to home page](index.html)

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
from PAMI.extras.DF2DB import DenseFormatDF as pro
import pandas as pd

# Objective: convert the above dataframe into a transactional database with items whose value is greater than or equal 1.
db = pro.DenseFormatDF(inputDF=pd.DataFrame('mentionDataFrame'), thresholdValue=1, condition='>=')
# Convert and store the dataframe as a transactional database file
db.createTransactional(outputFile='/home/userName/transactionalDB.txt')
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
from PAMI.extras.DF2DB import DenseFormatDF as pro
import pandas as pd

# Objective: convert the above dataframe into a transactional database with items whose value is greater than or equal 1.
db = pro.DenseFormatDF(inputDF=pd.DataFrame('mentionDataFrame'), thresholdValue=1, condition='>=')
# Convert and store the dataframe as a temporal database file
db.createTemporal(outputFile='/home/userName/temporalDB.txt')
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
from PAMI.extras.DF2DB import DenseFormatDF as pro
import pandas as pd

# Objective: convert the above dataframe into a utility database with items whose value is greater than or equal 1.
db = pro.DenseFormatDF(inputDF=pd.DataFrame('mentionDataFrame'), thresholdValue=1, condition='>=')
# Convert and store the dataframe as a utility database file
db.createUtility(outputFile='/home/userName/utilityDB.txt')
# Getting the fileName of the utility database
print('The output file is saved at ' + db.getFileName())

  ```

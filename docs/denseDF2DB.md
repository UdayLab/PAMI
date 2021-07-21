# Dense dataframe

### Introduction
A dense dataframe is basically a  matrix in which the first column represents the row-identifier/timestamp
and the remaining columns represent the values of the items. The format of a dense dataframe is as follows:

      rowIdentifier/timestamp   Item1   Item2   ... ItemN

An example of a dense dataframe generated from the customer purchase database is as follows:

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

**In the field of Big Data Analytics, a dense dataframe has a close resemblance to columnar databases.
MongoDB and HBASE are  the classic examples for columnar databases**

### Converting a dense dataframe into different database formats
Currently, PAMI supports converting a dataframe into a transactional database, temporal database, ond a utility database.
The users can avail this support by employing the functions available in **dataPreprocessing.dense2DB** class.  
We now present these three methods.

#### Converting a dense dataframe into a transactional database
A [transactional database](transactionalDatabase.html) represents a sparse and binary representation of items occurring in a dataframe. 
The steps to convert a dataframe into a transactional database is as follows:

1. Initialize the dense2DB class by passing the following three parameters: 
   1. inputDataFrame  - the dataframe that needs to converted into a database
   1. thresholdValue  - this value will be used to convert a non-binary data frame into a binary database
   1. condition       - The condition that needs to employed on the threshold value. Currently, the users can specify 
      the following six constraints: >, >=, <, <=, ==, and !=.

1. Call 'createTransactional(outputFileName)' method to store the dataframe as a transactional database.

A sample program to convert a dataframe into a transactional database and use it in a pattern mining algorithm, say FP-growth, is provided below

 ```Python
   from PAMI.dataProcessing import dense2DB as pro
   from PAMI.frequentPattern.basic import fpGrowth as alg
   import pandas as pd
   
   # Objective: convert the above dataframe into a transactional database with items whose value is greater than or equal 1.
   db=pro.dense2DB(inputDataFrame=pd.DataFrame('mentionDataFrame'), thresholdValue=1, condition='>=')
   # Convert and store the dataframe as a transactional database file
   db.createTransactional(outputFile='/home/userName/transactionalDB.txt')  
   # Getting the fileName of the transactional database
   print('The output file is saved at ' + db.getFileName())

   #Using the generated transactional database in FP-growth algorithm to discover frequent patterns

   obj = alg.fpGrowth(iFile=db.getFileName(), minSup='10.0')
   obj.startMine()
   patternsDF = obj.getPatternsInDataFrame()

   ```

#### Converting a dense dataframe into a temporal database
A [temporal database](temporalDatabase.html) represents a sparse and binary representation of items occurring at a particular timestamp
in a dataframe.  The steps to convert a dataframe into a temporal database is as follows:

1. Initialize the dense2DB class by passing the following three parameters: 
   1. inputDataFrame  - the dataframe that needs to converted into a database
   1. thresholdValue  - this value will be used to convert a non-binary data frame into a binary database
   1. condition       - The condition that needs to employed on the threshold value. Currently, the users can specify 
      the following six constraints: >, >=, <, <=, ==, and !=.

1. Call 'createTemporal(outputFileName)' method to store the dataframe as a temporal database.

A sample program to convert a dataframe into a temporal database and use it in a pattern mining algorithm, say PFP-growth++, is provided below

 ```Python
   from PAMI.dataProcessing import dense2DB as pro
   from PAMI.periodicFrequentPattern.basic import PFPGrowthPlus as alg
   import pandas as pd
   
   # Objective: convert the above dataframe into a transactional database with items whose value is greater than or equal 1.
   db=pro.dense2DB(inputDataFrame=pd.DataFrame('mentionDataFrame'), thresholdValue=1, condition='>=')
   # Convert and store the dataframe as a transactional database file
   db.createTransactional(outputFile='/home/userName/temporalDB.txt')  
   # Getting the fileName of the transactional database
   print('The output file is saved at ' + db.getFileName())

   obj = alg.PFPGrowthPlus(db.getFileName(), minSup="2", maxPer="6")
   obj.startMine()
   patternsDF = obj.getPatternsInDataFrame()
   
  ``` 
#### Converting a dense dataframe into a utility database
A [utility database](utilityDatabase.html) represents a sparse and non-binary representation of items occurring in
each row of a dataframe.  The steps to convert a dataframe into a utility database is as follows:

1. Initialize the dense2DB class by passing the following three parameters: 
   1. inputDataFrame  - the dataframe that needs to converted into a database
   1. thresholdValue  - this value will be used to convert a non-binary data frame into a binary database
   1. condition       - The condition that needs to employed on the threshold value. Currently, the users can specify 
      the following six constraints: >, >=, <, <=, ==, and !=.

1. Call 'createUtility(outputFileName)' method to store the dataframe as a temporal database.

A sample program to convert a dataframe into a utility database and use it in a pattern mining algorithm, say EFIM, is provided below

 ```Python
   from PAMI.dataProcessing import dense2DB as pro
   from PAMI.highUtilityPatterns.basic import EFIM as alg
   import pandas as pd
   
   # Objective: convert the above dataframe into a transactional database with items whose value is greater than or equal 1.
   db=pro.dense2DB(inputDataFrame=pd.DataFrame('mentionDataFrame'), thresholdValue=1, condition='>=')
   # Convert and store the dataframe as a transactional database file
   db.createTransactional(outputFile='/home/userName/utilityDB.txt')     
   # Getting the fileName of the transactional database
   print('The output file is saved at ' + db.getFileName())

  ```

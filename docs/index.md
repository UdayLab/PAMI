**[CLICK HERE](manual.html)** to access the PAMI manual.

# User Manual 
Key concepts in each link were briefly mentioned to save your valuable time. Click on the necessary links to know more.

1. [About PAMI: Motivation and the people who supported](aboutPAMI.html)
   
   PAMI is a PAttern MIning Python library to discover hidden patterns in Big Data.

2. [Installation/Update/uninstall PAMI](installation.html)
   
         pip install pami
   
3. [Organization of algorithms in PAMI](organization.html)
   
   The algorithms in PAMI are organized in a hierarchical structure as follows: 
   
        PAMI.theoriticalModel.basic/maximal/closed/topk.algorithmName
   
4. [Format to create different databases](createDatabases.html)
   
    1. [Transactional database](transactionalDatabase.html)
       
            format: item1<sep>item2<sep>...<sep>itemN
       
    1. [Temporal database](temporalDatabase.html)

            format: timestamp<sep>item1<sep>item2<sep>...<sep>itemN
    1. [Neighborhood database](neighborhoodDatabase.html)
            
            format: spatialItem1<sep>spatialItem3<sep>spatialItem10<sep>...
       
    1. [Utility database](utilityDatabase.html)
       
            format: item1<sep>...<sep>itemN:totalUtility:utilityItem1<sep>...<sep>utilityItemN
    
    Default separator used in PAMI is tab space. However, users can override the separator with their choice.
   
5. [Getting the statistics of databases](databaseStats.html)
   1. [Statistics of a transactional database](transactionalDatabaseStats.md)
   
   2. [statistics of a temporal database](temporalDatabaseStats.md)
6. [Converting Dataframes to Databases](dataFrameCoversio.html)

   1. [Format of dense dataframe]((denseDF2DB.html)) 
    
          tid/timestamp<sep>item1<sep>item2<sep>...<sep>itemN

   2. [Format of sparse dataframe]((sparseDF2DB.html)) 

          tid/timestamp<sep>item<sep>value

   3. [Dataframe to database conversion](denseDF2DB.html)
   
       This program creates a database by specifying a single condition and a threshold value for all items in a database.
   Code to convert a dataframe into a transactional database:

          from PAMI.DF2DB import DF2DB as pro
          db = pro.DF2DB(inputDataFrame, thresholdValue, condition, DFtype)
          # DFtype='sparse'  or 'dense'. Default type of an input dataframe is sparse
          db.createTransactional(outputFile)

   4. [Dataframe to database conversed advanced](DF2DBPlus.html)

      This program user can specify a different condition and a threshold value for each item in the dataframe. Code to convert a dataframe into a transactional database:
      
          from PAMI.DF2DB import DF2DBPlus as pro
          db = pro.DF2DBPlus(inputDataFrame, itemConditionValueDataFrame, DFtype)
          # DFtype='sparse'  or 'dense'. Default type of an input dataframe is sparse
          db.createTransactional(outputFile)

   5. [Spatiotemporal dataframe to databases](stDF2DB.html)
   
6. [Exceuting Algorithms in PAMI](utilization.html)    
   1. Importing PAMI algorithms into your program
   2. Executing PAMI algorithms directly on the terminal
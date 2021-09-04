**[CLICK HERE](index.html)** to access the PAMI manual.

# User Manual 
Key concepts in each link were briefly mentioned to save your valuable time. Click on the necessary links to know more.

1. [About PAMI: Motivation and the people who supported](aboutPAMI.html)
   
   PAMI is a PAttern MIning Python library to discover hidden patterns in Big Data.

2. [Installation/Update/uninstall PAMI](installation.html)
   
         pip install pami
   
3. [Organization of algorithms in PAMI](organization.html)
   
   The algorithms in PAMI are organized in a hierarchical structure as follows: 
   
        PAMI
          |-theoriticalModel
                   |-patternType (e.g., basic/maximal/closed/topk)
                          |-algorithmName
   
   An user can import a pattern mining algorithm using the following syntax:

       PAMI.theoriticalModel.basic/closed/maximal/topk import Algorithm as algo
    
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
        This program outputs the statistical details of a transactional database. It will also output the distribution of items' frequencies and transactional lengths.
        
          from dbStats import transactionalDatabaseStats as tds
          
          obj=tds.transactionalDatabaseStats(inputFile,sep)
          obj.run()
          
          #Getting basic stats of a database
          print("Database Size: " + obj.getDatabaseSize())
          print("Minimum transaction length:" + obj.getMinimumTransactionLength())
          print("Maximum transaction length:" + obj.getMaximumTransactionLength()) 
          print("Average transaction length:" + obj.getAverageTransactionLength())
          print("Standard deviation of transaction lengths:" + obj.getStandardDeviationTransactionLength())
          
          #Distribution of items' frequencies and transactional lengths
          itemFrequencies = obj.getItemFrequencies() #format: <item: freq>
          tranLenDistribution = obj.getTransanctionalLengthDistribution()  #format: <tranLength: freq>
          obj.storeInFile(itemFrequencies,outputFileName)
          obj.storeInFile(tranLenDistribution,outputFileName)        
          
   2. [Statistics of a temporal database](temporalDatabaseStats.md)
        This program outputs the statistical details of a temporal database. It will also output the distribution of items' frequencies, transactional lengths, and number of transactions occurring at each timestamp.
        
          from dbStats import temporalDatabaseStats as tds
          
          obj=tds.temporalDatabaseStats(inputFile,sep)
          obj.run()
          
          #Getting basic stats of a database
          print("Database Size: " + obj.getDatabaseSize())
          print("Minimum transaction length:" + obj.getMinimumTransactionLength())
          print("Maximum transaction length:" + obj.getMaximumTransactionLength()) 
          print("Average transaction length:" + obj.getAverageTransactionLength())
          print("Standard deviation of transaction lengths:" + obj.getStandardDeviationTransactionLength())
          print("Minimum period:" + obj.getMinimumPeriod())
          print("Maximum period:" + obj.getMaximumPeriod())
          print("Average period:" + obj.getAveragePeriod())
         
          #Distribution of items' frequencies, transactional lengths, and distribution of transactions per timestamp
          itemFrequencies = obj.getItemFrequencies() #format: <item: freq>
          tranLenDistribution = obj.getTransanctionalLengthDistribution()  #format: <tranLength: freq>
          transactionsPerTimestamp = obj.getNumberOfTransactionsPerTimestamp() #format: <timestamp: freq>
          obj.storeInFile(itemFrequencies,outputFileName)
          obj.storeInFile(tranLenDistribution,outputFileName) 
          obj.storeInFile(transactionsPerTimeStamp,outputFileName)
 
          
          
6. [Converting dataframes to databases](dataFrameCoversio.html)

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
   1. [Importing PAMI algorithms into your program](useAlgo.html)
   
          from PAMI.frequentPattern.basic import fpGrowth  as alg
          
          obj = alg.fpGrowth(inputFile,minSup,sep)
          obj.startMine()
          obj.storePatternsInFile('patterns.txt')
          df = obj.getPatternsInDataFrame()
          print('Runtime: ' + str(obj.getRuntime()))
          print('Memory: ' + str(obj.getMemoryRSS()))

   2. [Executing PAMI algorithms directly on the terminal](terminalExecute.html)
        1. Download the PAMI-main.zip file from  [GitHub](https://github.com/udayRage/PAMI/archive/refs/heads/main.zip)
        2. Unzip the PAMI-main.zip file.
        3. Enter into the PAMI-main folder and move the PAMI sub-folder to the location of your choice. 
        4. Let the location be /home/username/PAMI
        5. Execute the following command:
        
          python PAMI/patternModel/patternType/algorithm.py inputFile outputFile parameters
          
          E.g., python PAMI/frequentPattern/basic/fpGrowth.py inputFile outputFile minSup
7. [Extras](extras.html)
   1. [Creation of neighborhood file for spatiotemporal data using Euclidean distance](neighborhoodDatabase.md)
   
          from PAMI.extras import createNeighborhoodFileUsingEuclideanDistance as alg
          
          obj = alg.createNeighborhoodFileUsingEuclideanDistance(inputFile,outputFile,maxEuclideanDistance)
          obj.create()
          
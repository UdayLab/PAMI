# **[Home](index.html) | [Exercises](exercises.html) | [Real-world Examples](examples.html)**  

# User Manual 
Key concepts in each link were briefly mentioned to save your valuable time. Click on the necessary links to know more.

1. [About PAMI: Motivation and the people who supported](aboutPAMI.html)
   
   PAMI is a PAttern MIning Python library to discover hidden patterns in Big Data.

2. [Installation, update, and uninstallation of PAMI](installation.html)
   
   Install the PAMI library
   
         pip install pami
   
   Upgrade the PAMI library
   
        pip install --upgrade pami
        
    Uninstall the PAMI library
    
        pip uninstall -y pami
    
  
3. [Organization of algorithms in PAMI](organization.html)
   
   In PAMI, all algorithms were arranged in a hierarchical structure as follows: 
   
        PAMI
          |-theoriticalPatternModel
          |        |-patternType (e.g., basic/maximal/closed/topk)
          |               |-algorithmName
          |- ...
          |
          |-extras
                |-DF2DB                    
                |-dbStats
                |-plots
                |-...
   
   An user can import a pattern mining algorithm using the following syntax:

       PAMI.theoriticalPatternModel.<basic/closed/maximal/topk> import Algorithm as algo
    
4. [Format to create different types of databases](createDatabases.html)

   PAMI considers "tab space" as the default separator to distinguish items within a transaction. However, users can override this seperator.

   The algorithms in PAMI support both numeric and string data types. Thus, the items within a database can be numbers or strings.  

    1. [Transactional database](transactionalDatabase.html)
       
            format: item1<sep>item2<sep>...<sep>itemN
       
    2. [Temporal database](temporalDatabase.html)

            format: timestamp<sep>item1<sep>item2<sep>...<sep>itemN
    3. [Spatial database](spatialDatabase.html)
            
        Format of a spatio-transactional database is
        
            spatialItem1<sep>spatialItem2<sep> ... <sep>spatialItemN
       
       Format of a spatio-temporal database is
       
            timestamp<sep>spatialItem1<sep>spatialItem2<sep> ... <sep>spatialItemN
            
    4. [Utility database](utilityDatabase.html)
       
            format: item1<sep>...<sep>itemN:totalUtility:utilityItem1<sep>...<sep>utilityItemN
    
    5. [Uncertain database](uncertainDatabases.md)
            
       Uncertain transactional database

           format: item1<sep>...<sep>itemN:uncertainityValueItem1<sep>...<sep>uncertainityValueItemN

       Uncertain temporal database

           format: timestamp<sep>item1<sep>...<sep>itemN:uncertainityValueItem1<sep>...<sep>uncertainityValueItemN
5. [Converting dataframes to databases](dataFrameCoversio.html)

      Dataframes are popularly used in Python programing language for pipelining purposes. Depending on the arrangment 
      of the data in a data frame, we categorize them as a dense data frame and a sparse data frame. PAMI currently
provides in-built procedures to convert these data frames into transactional and temporal databases.

   1. [Format of dense dataframe]((denseDF2DB.html)) 
    
          tid/timestamp<sep>item1<sep>item2<sep>...<sep>itemN

   2. [Format of sparse dataframe]((sparseDF2DB.html)) 

          tid/timestamp<sep>item<sep>value

   3. [Basic approach to convert a dataframe into a database](denseDF2DB.html)
   
       This program creates a database by specifying a single condition and a threshold value for all items in a database.
   Code to convert a dataframe into a transactional database:

          from PAMI.extras.DF2DB import DF2DB as convertBasic
          
          db = convertBasic.DF2DB(inputDataFrame, thresholdValue, condition, DFtype)
          # DFtype='sparse'  or 'dense'. Default type of an input dataframe is sparse
   
          #Creates transactional database from a dataframe
          db.createTransactional(outputFileName)
   
          #Creates temporal database from a dataframe
          db.createTemporal(outputFileName) 
   
   4. [An advanced approach to convert a dataframe into a database](DF2DBPlus.html)

      The basic approach (mentioned above) allows the user to specify a single condition to all items in a data frame. However, in some real-world applications, users may need to 
      a different condition and a different threshold value for each item in a dataframe. The below code facilitates the user to create a database by specifying a different condition and a value for each item in a data frame.
      
          from PAMI.extras.DF2DB import DF2DBPlus as convertAdvanced
          
          db = convertAdvanced.DF2DBPlus(inputDataFrame, itemConditionValueDataFrame, DFtype)
          # DFtype='sparse'  or 'dense'. Default type of an input dataframe is sparse
          db.createTransactional(outputFile)

   5. [Spatiotemporal dataframe to databases](stDF2DB.html)
   
      To be written.
   
6. [Creation of synthetic databases](createDatabases.html)

    1. [Creation of transactional database](createTransactionalDatabase.html)  
   
       To be written.
    2. [Creation of temporal database](createTemporalDatabase.html) 
   
       To be written.
    3. [Creation of Spatiotemporal database](createSpatiotemporalDatabase.html)
   
       To be written.

7. [Understanding the statistics of a database](databaseStats.html)

    The performance of a mining algorithm primarily depends on the following two key factors: 

   1. Distribution of items' frequencies and 
   2. Distribution of transaction length

    Thus, it is important to know the statistical details of a database. PAMI provides inbuilt classes and functions to 
derive the statistical details of a database.
   
   1. [Statistics of a transactional database](transactionalDatabaseStats.md)
   
        This program outputs the statistical details of a transactional database. The details include distribution of items' frequencies and transactional lengths.
        
            import PAMI.extras.dbStats.transactionalDatabaseStats as tds
            
            obj = tds.transactionalDatabaseStats(inputFile)
            # obj = tds.transactionalDatabaseStats(inputFile, sep=',')  #overrride default tab seperator
            obj.run()
          
            print(f'Database size : {obj.getDatabaseSize()}')
            print(f'Total number of items : {obj.getTotalNumberOfItems()}')
            printf(f'Database sparsity : {obj.getSparsity()}')
            print(f'Minimum Transaction Size : {obj.getMinimumTransactionLength()}')
            print(f'Average Transaction Size : {obj.getAverageTransactionLength()}')
            print(f'Maximum Transaction Size : {obj.getMaximumTransactionLength()}')
            print(f'Standard Deviation Transaction Size : {obj.getStandardDeviationTransactionLength()}')
            print(f'Variance in Transaction Sizes : {obj.getVarianceTransactionLength()')
            
            itemFrequencies = obj.getSortedListOfItemFrequencies()
            transactionLength = obj.getTransanctionalLengthDistribution()
            obj.save(itemFrequencies, 'itemFrequency.csv')
            obj.save(transactionLength, 'transactionSize.csv')
     
   2. [Statistics of a temporal database](temporalDatabaseStats.md)
   
        This program outputs the statistical details of a temporal database. The details include distribution of items' frequencies, transactional lengths, and number of transactions occurring at each timestamp.
        
            import PAMI.extras.dbStats.temporalDatabaseStats as tds
          
            obj = tds.temporalDatabaseStats(inputFile)
            # obj = tds.temporalDatabaseStats(inputFile, sep=',')  #overrride default tab seperator
            obj.run()
            
            print(f'Database size : {obj.getDatabaseSize()}')
            print(f'Total number of items : {obj.getTotalNumberOfItems()}')
            printf(f'Database sparsity : {obj.getSparsity()}')
            print(f'Minimum Transaction Size : {obj.getMinimumTransactionLength()}')
            print(f'Average Transaction Size : {obj.getAverageTransactionLength()}')
            print(f'Maximum Transaction Size : {obj.getMaximumTransactionLength()}')
            print(f'Standard Deviation Transaction Size : {obj.getStandardDeviationTransactionLength()}')
            print(f'Variance in Transaction Sizes : {obj. getVarianceTransactionLength()')           
            
            print(f'Minimum period : {obj.getMinimumPeriod()}')
            print(f'Average period : {obj.getAveragePeriod()}')
            print(f'Maximum period : {obj.getMaximumPeriod()}')
            
            itemFrequencies = obj.getSortedListOfItemFrequencies()
            transactionLength = obj.getTransanctionalLengthDistribution()
            numberOfTransactionPerTimeStamp = obj.getNumberOfTransactionsPerTimestamp()
            obj.save(itemFrequencies,'itemFrequency.csv')
            obj.save(transactionLength, 'transactionSize.csv')
            obj.save(numberOfTransactionPerTimeStamp, 'numberOfTransaction.csv')
 
   3. [Statistics of a utility database](utilityDatabaseStats.md)
   
        This program outputs the statistical details of a utility database. It will also output the distribution of items' frequencies, transactional lengths, and sum of utilities of all items in a database.
       
            import PAMI.extras.dbStats.utilityDatabaseStats as uds
            
            obj = uds.utilityDatabaseStats(inputFile)
            #obj = uds.utilityDatabaseStats(inputFile,sep=',') #override default tab separator
            obj.run()
            
            print(f'Database size : {obj.getDatabaseSize()}')
            print(f'Total number of items : {obj.getTotalNumberOfItems()}')
            printf(f'Database sparsity : {obj.getSparsity()}')
            print(f'Minimum Transaction Size : {obj.getMinimumTransactionLength()}')
            print(f'Average Transaction Size : {obj.getAverageTransactionLength()}')
            print(f'Maximum Transaction Size : {obj.getMaximumTransactionLength()}')
            print(f'Standard Deviation Transaction Size : {obj.getStandardDeviationTransactionLength()}')
            print(f'Variance in Transaction Sizes : {obj. getVarianceTransactionLength()')
            print(f'Total utility : {obj.getTotalUtility()}')
            print(f'Minimum utility : {obj.getMinimumUtility()}')
            print(f'Average utility : {obj.getAverageUtility()}')
            print(f'Maximum utility : {obj.getMaximumUtility()}')
            itemFrequencies = obj.getSortedListOfItemFrequencies()
            transactionLength = obj.getTransanctionalLengthDistribution()
            utility = obj.getSortedUtilityValuesOfItem()
            obj.save(itemFrequencies, 'itemFrequency.csv')
            obj.save(transactionLength, 'transactionSize.csv')
            obj.save(utility, 'utility.csv')            
   
8. [Basic plots of a database](basicPlots.md)

    In the previous chapter, we have presented the methods to derive the statistics of a database. 
    In continuation, we present the methods to plot the graphs. 
    
        import PAMI.extras.graph.plotLineGraphFromDictionary as plt
        
        plt.plotLineGraphFromDictionary(dictionary,percentageOfItemsToPlot,title,xLabel,yLabel) 
     
   Example: __Drawing plots for a transactional database.__
   
          import PAMI.extras.graph.plotLineGraphFromDictionary as plt
          
          obj = tds.transactionalDatabaseStats(inputFile)
          # obj = tds.transactionalDatabaseStats(inputFile, sep=',')  #overrride default tab seperator
          obj.run()
          
          import PAMI.extras.graph.plotLineGraphFromDictionary as plt   
          plt.plotLineGraphFromDictionary(obj.getSortedListOfItemFrequencies(),50,'item frequencies', 'item rank', 'frequency')
          plt.plotLineGraphFromDictionary(obj.getTransanctionalLengthDistribution(),100,'distribution of transactions', 'transaction length', 'frequency') 

9. [Exceuting Algorithms in PAMI](utilization.html)

   1. [Importing PAMI algorithms into your program](useAlgo.html)
   
          from PAMI.frequentPattern.basic import fpGrowth  as alg
          
          obj = alg.fpGrowth(inputFile,minSup,sep)
          obj.startMine()
          obj.save('patterns.txt')
          df = obj.getPatternsAsDataFrame()
          print('Runtime: ' + str(obj.getRuntime()))
          print('Memory: ' + str(obj.getMemoryRSS()))

   2. [Executing PAMI algorithms directly on the terminal](terminalExecute.html)
        1. Download the PAMI-main.zip file from [GitHub](https://github.com/udayRage/PAMI/archive/refs/heads/main.zip)
        2. Unzip the PAMI-main.zip file.
        3. Enter into the PAMI-main folder and move the PAMI sub-folder to the location of your choice. 
        4. Let the location be /home/username/PAMI
        5. Execute the following command:
        
        
          python PAMI/patternModel/patternType/algorithm.py inputFile outputFile parameters
          
          E.g., python PAMI/frequentPattern/basic/fpGrowth.py inputFile outputFile minSup

9. [Extras](extras.html)
    1. [Generating latex graphs for publishing results in conferences and journals](generateLatexGraphs.html)
   
         To be written.
    2. [Creation of neighborhood file from spatiotemporal database using Euclidean distance](spatialDatabase.md)
   
           from PAMI.extras.neighbours import createNeighborhoodFileUsingEuclideanDistance as alg
          
           obj = alg.createNeighborhoodFileUsingEuclideanDistance(inputFile,outputFile,maxEuclideanDistance,seperator) 
          
# **[Home](index.html) | [Exercises](exercises.html) | [Real-world Examples](examples.html)**  

# User Manual 
Key concepts in each link were briefly mentioned to save your valuable time. Click on the necessary links to know more.

1. [About PAMI: Motivation and the people who supported](aboutPAMI.html)
   
   PAMI is a PAttern MIning Python library to discover hidden patterns in Big Data.

2. [Installation/Update/uninstall PAMI](installation.html)
   
   Install the PAMI library
   
         pip install pami
   
   Upgrade the PAMI library
   
        pip install --upgrade pami
        
    Uninstall the PAMI library
    
        pip uninstall -y pami
        
3. [Organization of algorithms in PAMI](organization.html)
   
   The algorithms in PAMI are organized in an hierarchical structure as follows: 
   
        PAMI
          |-theoriticalModel
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

       PAMI.theoriticalModel.basic/closed/maximal/topk import Algorithm as algo
    
4. [Format to create different databases](createDatabases.html)
   
    1. [Transactional database](transactionalDatabase.html)
       
            format: item1<sep>item2<sep>...<sep>itemN
       
    1. [Temporal database](temporalDatabase.html)

            format: timestamp<sep>item1<sep>item2<sep>...<sep>itemN
    1. [Spatial database](spatialDatabase.html)
            
        Format of a spatio-transactional database is
        
            spatialItem1<sep>spatialItem2<sep> ... <sep>spatialItemN
       
       Format of a spatio-temporal database is
       
            timestamp<sep>spatialItem1<sep>spatialItem2<sep> ... <sep>spatialItemN
            
    1. [Utility database](utilityDatabase.html)
       
            format: item1<sep>...<sep>itemN:totalUtility:utilityItem1<sep>...<sep>utilityItemN
    
    Default separator used in PAMI is tab space. However, users can override the separator with their choice.
   
5. [Getting the statistics of databases](databaseStats.html)

The performance of a mining algorithm primarily depends on the following two key factors: 
1. Distribution of items' frequencies and 
1. Distribution of transaction length

Thus, it is important to know the statistical details of a database. PAMI provides inbuilt classes and functions methods to 
get the statistical details of a database.
   
   1. [Statistics of a transactional database](transactionalDatabaseStats.md)
   
        This program outputs the statistical details of a transactional database. It will also output the distribution of items' frequencies and transactional lengths.
        
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
            obj.storeInFile(itemFrequencies, 'itemFrequency.csv')
            obj.storeInFile(transactionLength, 'transactionSize.csv')        
          
   2. [Statistics of a temporal database](temporalDatabaseStats.md)
   
        This program outputs the statistical details of a temporal database. It will also output the distribution of items' frequencies, transactional lengths, and number of transactions occurring at each timestamp.
        
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
            obj.storeInFile(itemFrequencies,'itemFrequency.csv')
            obj.storeInFile(transactionLength, 'transactionSize.csv')
            obj.storeInFile(numberOfTransactionPerTimeStamp, 'numberOfTransaction.csv')
 
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
            obj.storeInFile(itemFrequencies, 'itemFrequency.csv')
            obj.storeInFile(transactionLength, 'transactionSize.csv')
            obj.storeInFile(utility, 'utility.csv')            
   
6. [Basic plots of a database](basicPlots.md)

    In the previous chapter, we have presented the methods to understand the statistics of a database. 
    In continuation, we present the methods to plot the graphs of graphs. 
    
        import PAMI.extras.graph.plotLineGraphFromDictionary as plt
        
        plt.plotLineGraphFromDictionary(dictionary,percentageOfItemsToPlot,title,xLabel,yLabel) 
     
   1. [Plotting items' frequencies](plotItemFrequencies.md)
   
          import PAMI.extras.graph.plotLineGraphFromDictionary as plt
          
          obj = tds.transactionalDatabaseStats(inputFile)
          # obj = tds.transactionalDatabaseStats(inputFile, sep=',')  #overrride default tab seperator
          obj.run()
           
          plt.plotLineGraphFromDictionary(obj.getSortedListOfItemFrequencies(),percentageOfItemsToPlot,title,xLabel,yLabel) 
        
   2. [Plotting distribution of transaction lengths](plotTransactionDistribution.md)
   
          import PAMI.extras.graph.plotLineGraphFromDictionary as plt
          
          obj = tds.transactionalDatabaseStats(inputFile)
          # obj = tds.transactionalDatabaseStats(inputFile, sep=',')  #overrride default tab seperator
          obj.run()
           
          plt.plotLineGraphFromDictionary(obj.getTransanctionalLengthDistribution(),percentageOfItemsToPlot,title,xLabel,yLabel) 
                  
7. [Converting dataframes to databases](dataFrameCoversio.html)

   1. [Format of dense dataframe]((denseDF2DB.html)) 
    
          tid/timestamp<sep>item1<sep>item2<sep>...<sep>itemN

   2. [Format of sparse dataframe]((sparseDF2DB.html)) 

          tid/timestamp<sep>item<sep>value

   3. [Dataframe to database conversion](denseDF2DB.html)
   
       This program creates a database by specifying a single condition and a threshold value for all items in a database.
   Code to convert a dataframe into a transactional database:

          from PAMI.extras.DF2DB import DF2DB as pro
          
          db = pro.DF2DB(inputDataFrame, thresholdValue, condition, DFtype)
          # DFtype='sparse'  or 'dense'. Default type of an input dataframe is sparse
          db.createTransactional(outputFile)

   4. [Dataframe to database conversed advanced](DF2DBPlus.html)

      This program user can specify a different condition and a threshold value for each item in the dataframe. Code to convert a dataframe into a transactional database:
      
          from PAMI.extras.DF2DB import DF2DBPlus as pro
          
          db = pro.DF2DBPlus(inputDataFrame, itemConditionValueDataFrame, DFtype)
          # DFtype='sparse'  or 'dense'. Default type of an input dataframe is sparse
          db.createTransactional(outputFile)

   5. [Spatiotemporal dataframe to databases](stDF2DB.html)
   
6. [Exceuting Algorithms in PAMI](utilization.html)    
   1. [Importing PAMI algorithms into your program](useAlgo.html)
   
          from PAMI.frequentPattern.basic import fpGrowth  as alg
          
          obj = alg.fpGrowth(inputFile,minSup,sep)
          obj.startMine()
          obj.savePatterns('patterns.txt')
          df = obj.getPatternsAsDataFrame()
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
   1. [Creation of neighborhood file for spatiotemporal data using Euclidean distance](spatialDatabase.md)
   
          from PAMI.extras import findNeighboursUsingEuclidean as alg
          
          obj = alg.findNeighboursUsingEuclidean(inputFile,outputFile,maxEuclideanDistance)
          obj.create()
          
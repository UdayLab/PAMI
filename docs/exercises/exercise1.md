# **[Home](index.html) | [Exercises](exercises.html) | [Real-world Examples](examples.html)**  

# Exercise 1: Mining frequent patterns in very large databases using a single algorithm

Tasks in this exercise:

- [ ] Creation of synthetic database
- [ ] Understanding the characteristics of the database
- [ ] Implementing Apriori algorithm
- [ ] Visualizing the results of Apriori algorithm


## Task 1: Creation of synthetic database

A synthetic transactional database can be created by calling `generateTransactionalDatabase` class in PAMI.extras.generateDatabase. 
A sample code to create a synthetic database is as follows:

    import PAMI.extras.generateDatabase.generateTransactionalDatabase as dbGenerator

    totalNumberOfItems=500      #total number of items that must exist in a database. Symbol used for this term is I
    totalNumberOfTransactions=1000     #Number of transactions that must exist in a database. Symbol used for this term is D
    probabilityOfOccurrenceOfAnItem=20  #The probability with which an item must occur in a transaction. The value ranges between 0 to 100. Symbol used for this term is P 

    outputFile='D1000I500P20.tsv'
    data = dbGenerator.generateTransactionalDatabase(totalNumberOfItems, totalNumberOfTransactions, probabilityOfOccurrenceOfAnItem, outputFile)
 



[Click here](../createTransactionalDatabase.html) to know more about `generateTransactionalDatabase` class.

## Task 2: Understanding the characteristics of the database

 Statistical details of a transactional database can be obtained by calling `transactionalDatabaseStats` class in PAMI.extras.dbStats.
 These details also be visualized by calling `plotLineGraphFromDictionary` class in  PAMI.extras.graph.

    import PAMI.extras.dbStats.transactionalDatabaseStats as stats
    import PAMI.extras.graph.plotLineGraphFromDictionary as plt 
            
    obj = stats.transactionalDatabaseStats(inputFile) 
    obj.run()
  
    print(f'Database size : {obj.getDatabaseSize()}')
    print(f'Total number of items : {obj.getTotalNumberOfItems()}')
    print(f'Database sparsity : {obj.getSparsity()}')
    print(f'Minimum Transaction Size : {obj.getMinimumTransactionLength()}')
    print(f'Average Transaction Size : {obj.getAverageTransactionLength()}')
    print(f'Maximum Transaction Size : {obj.getMaximumTransactionLength()}')
    print(f'Standard Deviation Transaction Size : {obj.getStandardDeviationTransactionLength()}')
    print(f'Variance in Transaction Sizes : {obj.getVarianceTransactionLength()}')
    
    itemFrequencies = obj.getSortedListOfItemFrequencies()
    transactionLength = obj.getTransanctionalLengthDistribution()
    obj.storeInFile(itemFrequencies, 'itemFrequency.csv')
    obj.storeInFile(transactionLength, 'transactionSize.csv')

    #print top 50% of the frequent items
    plt.plotLineGraphFromDictionary(obj.getSortedListOfItemFrequencies(),50,'item frequencies', 'item rank', 'frequency')

    
    plt.plotLineGraphFromDictionary(obj.getTransanctionalLengthDistribution(),100,'distribution of transactions', 'transaction length', 'frequency') 

[Click here](../transactionalDatabaseStats.html) to know more about `transactionalDatabaseStats` class.

[Click here](../basicPlots.html) to know more about `plotLineGraphFromDictionary` class.

## Task 3:  Implementing Apriori algorithm
For the purpose of brevity, we first discuss how to execute Apriori algorithm on a database at a particular minimum support value. 
In the next step, we generalize the above step by executing Apriori algorithm at different minimum support values.

### Task 3.1: Implementing Apriori algorithm at a particular minSup
 
The Apriori algorithm can be executed by calling `Apriori` class in  PAMI.frequentPattern.basic. 

    from PAMI.frequentPattern.basic import Apriori  as alg
          
    minSup=0.8     #minimum support of a pattern
    sep='\t'       #default seperator used to seperate items in a database

    obj = alg.Apriori(inputFile,minSup,sep)
    obj.startMine()

    obj.savePatterns('patterns.txt')
    df = obj.getPatternsAsDataFrame()
    print('Total number of patterns:', len(df))
    print('Runtime: ' + str(obj.getRuntime()))
    print('Memory: ' + str(obj.getMemoryRSS()))

### Task 3.2: Implementing Apriori algorithm by varying minSup

The Apriori algorithm can be executed by calling `Apriori` class in  PAMI.frequentPattern.basic. 

    import pandas as pd
    result = pd.DataFrame(columns=['algorithm', 'minSup', 'patterns', 'runtime', 'memory'])  # To store the output in dataframe format.

    from PAMI.frequentPattern.basic import Apriori as alg
    
    minSupList = [0.5, 0.6, 0.7, 0.8, 0.9]
    sep = '\t'
    
    algorithm = 'Apriori'
    numOfPatterns = {}
    runtime = {}
    memoryUSS = {}
    memoryRSS = {}
    for minSup in minSupList:
        obj = alg.Apriori(inputFile, minSup=minSup, sep=sep)
        obj.startMine()
        numOfPatterns[minSup]  = obj.getPatterns()
        runtime[minSup] = obj.getRuntime()
        memoryUSS[minSup] = obj.getMemoryUSS()
        memoryRSS[minSup] = obj.getMemoryRSS()

        
    

## Task 4: Visualizing the results of Apriori algorithm

The visualization is done using the dataframe (result) generated by running the algorithm for multiple minSup above.

    from PAMI.extras.graph import plotLineFromDictionary as plt
    plt.plotLineGraphFromDictionary(numOfPatterns, 100, title = 'patterns', xlabel = 'minSup', ylabel='no of patterns')
    plt.plotLineGraphFromDictionary(runtime, 100, title = 'runtime', xlabel = 'minSup', ylabel='runtime')
    plt.plotLineGraphFromDictionary(memoryUSS, 100, title = 'memory', xlabel = 'minSup', ylabel='memory')
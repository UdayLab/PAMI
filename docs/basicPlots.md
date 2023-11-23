# **[Home](index.html) | [Exercises](exercises.html) | [Real-world Examples](examples.html)**  

#Drawing line graphs from a dictionary

Choosing an appropriate threshold value(s) is a challenging task in pattern mining. This challenge can be addressed by understanding
the distribution of characteristics of items and transactions. PAMI provides an excellent interface to view the distributions
of the items and transactions in a database. The code to draw graphs for a dictionary is as follows:


    import PAMI.extras.graph.plotLineGraphFromDictionary as plt
        
    plt.plotLineGraphFromDictionary(dictionary,percentageOfItemsToPlot,title,xLabel,yLabel) 


Input parameters:

1. **dictionary** - dictionary with key:value pairs. The dictionary is assumed to be sorted in an order (ascending or descending)
2. **percentageOfItemsToPlot** - this parameter controls the number of elements in the dictionary that needs to be drawn in the graph. The default value is 100% percentage. That is, draw the graph with all the elements in the dictionary.
3. **title** - title to be displayed on the graph. The default value is empty string.
4. **xlabel** - text to be displayed below x-axis. The default value is empty string.
5. **ylabel** -text to be display below y-axis. The default value is empty string.

## Plots for a transactional database

    import PAMI.extras.dbStats.TransactionalDatabase as tds
    import PAMI.extras.graph.plotLineGraphFromDictionary as plt
            
    obj = tds.TransactionalDatabase(inputFile)
    # obj = tds.TransactionalDatabase(inputFile, sep=',')  #overrride default tab seperator
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

    plt.plotLineGraphFromDictionary(itemFrequencies,50,'item frequencies', 'item rank', 'frequency') 
    #above command plots the graph with top 50% of the elements in the dictionary
    plt.plotLineGraphFromDictionary(transactionLength,100,'distribution of transactions', 'transaction length', 'frequency') 
    #above command plots the graph with top 100% of the elements in the dictionary

## Plots for a temporal database

    import PAMI.extras.dbStats.temporalDatabaseStats as tds
    import PAMI.extras.graph.plotLineGraphFromDictionary as plt
          
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

    plt.plotLineGraphFromDictionary(itemFrequencies,50,'item frequencies', 'item rank', 'frequency') 
    #above command plots the graph with top 50% of the elements in the dictionary
    plt.plotLineGraphFromDictionary(transactionLength,100,'distribution of transactions', 'transaction length', 'frequency') 
    #above command plots the graph with top 100% of the elements in the dictionary
    plt.plotLineGraphFromDictionary(numberOfTransactionPerTimeStamp,100,'distribution of timestamps', 'timestamps', 'frequency') 
    #above command plots the graph with top 100% of the elements in the dictionary         

## Plots for a utility database

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

    plt.plotLineGraphFromDictionary(itemFrequencies,50,'item frequencies', 'item rank', 'frequency') 
    #above command plots the graph with top 50% of the elements in the dictionary
    plt.plotLineGraphFromDictionary(transactionLength,100,'distribution of transactions', 'transaction length', 'frequency') 
    #above command plots the graph with top 100% of the elements in the dictionary
    plt.plotLineGraphFromDictionary(utility,100,'distribution of utility values', 'item rank', 'utility') 
    #above command plots the graph with top 100% of the elements in the dictionary 
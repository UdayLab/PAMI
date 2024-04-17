[Previous](TransactionalDatabase.html)|[🏠 Home](index.html)|[Next](utilityDatabaseStats.html)

## Statistical details of a temporal database

The performance of a mining algorithm primarily depends on the following two key factors: 
1. Distribution of items' frequencies and 
1. Distribution of transaction length

Thus, it is important to know the statistical details of a database. PAMI provides inbuilt classes and functions methods to 
get the statistical details of a database.   In this page, we provide the details of methods to get statistical details from 
a temporal database. 

### Executing TemporalDatabase program

The TemporalDatabase.py program is located in PAMI.extras.dbStats folder. Thus, execute the below lines to run the program.

```Python
    # import the program
import PAMI.extras.dbStats.TemporalDatabase as tds

inputFile = "fileName"
# initialize the program
obj = tds.TemporalDatabase(inputFile)
# obj = tds.TemporalDatabase(inputFile, sep=',')  #overrride default tab seperator
# execute the program
obj.run()
```
Once the program is executed, users can call different methods to get the statistical details of a database. We now describe the available methods.

#### getDatabaseSize()
    
   This method returns the total number of transactions in a database.  
```Python  
    print(f'Database size : {obj.getDatabaseSize()}')
```

#### getTotalNumberOfItems()

   This method returns the total number of transactions in a database.
```Python 
    print(f'Total number of items : {obj.getTotalNumberOfItems()}')
```
####.getSparsity()    

   This method returns the sparsity (i.e., the portion of empty values) of the database.
```Python   
    print(f'Database sparsity : {obj.getSparsity()}')
```
#### getMinimumTransactionLength()

   This method  returns the length of the small transaction in a database. In other words, this function returns the minimum number of items in a transaction.
```Python   
    print(f'Minimum Transaction Size : {obj.getMinimumTransactionLength()}')
```
#### getAverageTransactionLength()

   This method  returns the length of an average transaction in a database. In other words, this function returns the average number of items in a transaction.
```Python   
    print(f'Average Transaction Size : {obj.getAverageTransactionLength()}')
```   
#### getMaximumTransactionLength()
   This method returns the length of the largest transaction in a database. In other words, this function returns the maximum number of items in a transaction.

```Python
   print(f'Maximum Transaction Size : {obj.getMaximumTransactionLength()}')
```    
#### getStandardDeviationTransactionLength()
   This method returns the standard deviation of the lengths of transactions in database.
```Python
    print(f'Standard Deviation Transaction Size : {obj.getStandardDeviationTransactionLength()}')
```
#### getVarianceTransactionLength()

   This method returns the variance of the lengths of transactions in a database
```Python
    print(f'Variance in Transaction Sizes : {obj.getVarianceTransactionLength()')
```    
#### getVarianceTransactionLength()
   This method returns the varience of the lengths of transactions in database.
```Python
     print(f'Variance of Transaction Size :{obj.getVarianceTransactionLength()}')
```
#### getSparsity():
   This method returns the sparsity of the database.
```Python
    print(f'Database sparsity :{obj.getSparsity()}')
```
#### getMinimumPeriod()
   This method returns the minimum period between two transactions in a database.
```Python   
    print(f'Minimum period : {obj.getMinimumPeriod()}')
```
#### getAveragePeriod()
   This method returns the average period between two transactions in a database.
```Python    
     print(f'Average period : {obj.getAveragePeriod()}')
```     
#### getMaximumPeriod()
   This method returns the maximum period between two transactions in a database.
```Python   
      print(f'Maximum period : {obj.getMaximumPeriod()}')
```      
      
#### getSortedListOfItemFrequencies()
   This method returns a sorted dictionary of items and their frequencies in the database. The format of this dictionary is {item:frequency} 
   The items in this dictionary are sorted in frequency descending order. 
 ```Python  
    itemFrequencies = obj.getSortedListOfItemFrequencies()
```
#### getTransanctionalLengthDistribution()
   This method returns a sorted dictionary of transaction lengths and their occurrence frequencies in the database. 
   The format of this dictionary is {temporalLength:frequency}.
   The transaction lengths in this dictionary are sorted in ascending order of their temporal lengths.
```Python   
    transactionLength = obj.getTransanctionalLengthDistribution()
```
#### getNumberOfTransactionsPerTimestamp()
   This method returns a sorted dictionary of timestamps and the number of transactions occurring at the corresponding timestamp.
   The format of this dictionary is {timestamp:frequency}
```Python   
     numberOfTransactionPerTimeStamp = obj.getNumberOfTransactionsPerTimestamp()
```          
#### save(dictionary, returnFileName)
   This method stores the dictionary in a file. In the output file, the key value pairs of the dictionary are separated by a tab space. 
```Python   
    obj.save(itemFrequencies, 'itemFrequency.csv')
    obj.save(transactionLength, 'transactionSize.csv')       
    obj.save(numberOfTransactionPerTimeStamp, 'numberOfTransaction.csv')
```    
    
## Sample code 
```Python
    import PAMI.extras.dbStats.TemporalDatabase as tds
          
    obj = tds.TemporalDatabase(inputFile)
    # obj = tds.TemporalDatabase(inputFile, sep=',')  #overrride default tab seperator
    obj.run()
    
    print(f'Database size : {obj.getDatabaseSize()}')
    print(f'Total number of items : {obj.getTotalNumberOfItems()}')
    print(f'Database sparsity : {obj.getSparsity()}')
    print(f'Minimum Transaction Size : {obj.getMinimumTransactionLength()}')
    print(f'Average Transaction Size : {obj.getAverageTransactionLength()}')
    print(f'Maximum Transaction Size : {obj.getMaximumTransactionLength()}')
    print(f'Standard Deviation Transaction Size : {obj.getStandardDeviationTransactionLength()}')
    print(f'Variance in Transaction Sizes : {obj. getVarianceTransactionLength()}')
    print(f'Minimum period : {obj.getMinimumPeriod()}')
    print(f'Average period : {obj.getAveragePeriod()}')
    print(f'Maximum period : {obj.getMaximumPeriod()}')
    
    itemFrequencies = obj.getSortedListOfItemFrequencies()
    transactionLength = obj.getTransanctionalLengthDistribution()
    numberOfTransactionPerTimeStamp = obj.getNumberOfTransactionsPerTimestamp()
    obj.save(itemFrequencies,'itemFrequency.csv')
    obj.save(transactionLength, 'transactionSize.csv')
    obj.save(numberOfTransactionPerTimeStamp, 'numberOfTransaction.csv')
```Python






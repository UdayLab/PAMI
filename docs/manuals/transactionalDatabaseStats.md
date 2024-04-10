[__<--__ Previous ](aboutPAMI.html)|[Home](installation.html)|[_Next_-->](organization.html)

## Knowing the statistical details of a transactional database

The performance of a mining algorithm primarily depends on the following two key factors: 
1. Distribution of items' frequencies and 
1. Distribution of transaction length

Thus, it is important to know the statistical details of a database. PAMI contains inbuilt classes and functions methods to 
get the statistical details of a database.   In this page, we provide the details of methods to get statistical details of 
a transactional database. 

## Class name
The class to print the statistics of a transactional base is "_TransactionalDatabase_". This class is located at PAMI/extras/dbStats directory.
One can import this class using the following syntax: 

    import PAMI.extras.dbStats.TransactionalDatabase as stats

##Methods

1. __getDatabaseSize()__: This method returns the total number of transactions in a database.
2. __getTotalNumberOfItems()__:  This method returns the total number of transactions in a database.
3. __getSparsity()__: This method returns the sparsity (i.e., the portion of empty values) of the database.
4. __getMinimumTransactionLength()__: This method  returns the length of the small transaction in a database. In other words, this function returns the minimum number of items in a transaction.
5. __getAverageTransactionLength()__: This method  returns the length of an average transaction in a database. In other words, this function returns the average number of items in a transaction.
6. __getMaximumTransactionLength()__: This method returns the length of the largest transaction in a database. In other words, this function returns the maximum number of items in a transaction.
7. __getStandardDeviationTransactionLength()__: This method returns the standard deviation of the lengths of transactions in database.
8. __getVarianceTransactionLength()__: This method returns the variance of the lengths of transactions in a database.
9. __getSparsity()__: This method retuns the sparsity of the database.
10. __getSortedListOfItemFrequencies()__: This method returns a sorted dictionary of items and their frequencies in the database. The format of this dictionary is {item:frequency}. 
   The items in this dictionary are sorted in frequency descending order.
11. __getTransanctionalLengthDistribution():__ This method returns a sorted dictionary of transaction lengths and their occurrence frequencies in the database.  The format of this dictionary is {transactionalLength:frequency}.    The transaction lengths in this dictionary are sorted in ascending order of their transactional lengths.
12. __save(dictionary, outputFileName)__: This method stores the dictionary in a file. In the output file, the key value pairs of the dictionary are separated by a tab space.
13. __printStats()__: This method prints the complete statistical details of a database.
14. __plotGraphs()__: This method draws the graphs containing the statistical details of a database.

## Sample code  - 1: Printing all statistics

__Note:__ [Click here to download the transactional database](https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/Transactional_T10I4D100K.csv)

```Python
# import the class file
import PAMI.extras.dbStats.TransactionalDatabase as stats

# specify the file name
inputFile = 'Transactional_T10I4D100K.csv'

obj = stats.TransactionalDatabase(inputFile, sep='\t')
obj.run()

obj.printStats()
obj.plotGraphs()
```

## Sample code  - 2: Printing individual statistics

```Python
import PAMI.extras.dbStats.TransactionalDatabase as tds

inputFile = "<provide the name of a transactional database>"

# initialize the program
obj = tds.TransactionalDatabase(inputFile)
# obj = tds.TransactionalDatabase(inputFile,sep='\t') #override default tab seperator

# execute the program
obj.run()

# print the database stats
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
obj.save(itemFrequencies, 'itemFrequency.csv')
obj.save(transactionLength, 'transactionSize.csv')   
 ```

[Previous](createTransactionalDatabase.html)|[Home](index.html)|[Next](createSpatiotemporalDatabase.html)

## Creation of temporal database

This page describes the process to create synthetic transactional databases of varying sizes. __Please note that this code is different from the widely used 
synthetic IBM data generator.__
[__<--__ Return to home page](index.html)

## Creation of temporal databases

This page describes the process to create synthetic temporal databases of varying sizes. The users
can create different types of temporal databases.

- __Regular temporal database:__ A temporal database in which the inter-arrival time between the transactions is contact. 
- __Irregular temporal database (Type-I):__ A temporal database in which transactions appear at irregular time intervals. Please note that in this database, every transaction has a distinct timestamp. 
- __Irregular temporal database (Type-II):__ A temporal database in which multiple transactions share a common timestamp. 

__Please note that all of the above forms of temporal databases can be created by varying the input parameters.__

### Step 1: Import the program
A synthetic transactional database can be created by calling `generateTransactionalDatabase` class in PAMI.extras.generateDatabase.

```Python
import PAMI.extras.generateDatabase.generateTemporalDatabase as dbGenerator
```

### Step 2: Specify the parameters

```Python
totalNumberOfTransactions_D=1000     #Number of transactions that must exist in a database. Symbol used for this term is D
totalNumberOfItems_I=500      #total number of items that must exist in a database. Symbol used for this term is I
maximumLengthOfTransaction_T=20  #Maximum number of items that must exist in a database
probabilityOfRecurrenceOfTimeStamp_P=0   #Set the above value to 0 to create an irregular temporal database of Type-II.

sep = '\t'    "Specify the seperator. \t is the default seperator"
outputFile='D1000I500T20P0.tsv'   #Specify the file name. 
# 'D' represents the database size, 
# 'I' represents the total number of items and 
# 'P' represents the probability of occurrence of an item in a database
```
### Step 3: Creating the synthetic dataset

```Python
temporalDB = generateTemporalDatabase(totalNumberOfTransactions, totalNumberOfItems,temporalDB = generateTemporalDatabase(numOfTransactions, maxNumOfItems, maxNumOfItemsPerTransaction, outFileName, percent, sep)


temporalDB.createTemporalFile()
```

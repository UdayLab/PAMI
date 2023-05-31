[Return to home page](index.html)  

## Creation of transactional database

This page describes the process to create synthetic transactional databases of varying sizes. __Please note that this code is different from the widely used 
synthetic IBM data generator.__

### Step 1: Import the program
A synthetic transactional database can be created by calling `generateTransactionalDatabase` class in PAMI.extras.generateDatabase. 

```Python
import PAMI.extras.generateDatabase.generateTransactionalDatabase as dbGenerator
```
### Step 2: Specify the parameters
```Python
totalNumberOfItems=500      #total number of items that must exist in a database. Symbol used for this term is I
totalNumberOfTransactions=1000     #Number of transactions that must exist in a database. Symbol used for this term is D
probabilityOfOccurrenceOfAnItem=20  #The probability with which an item must occur in a transaction. The value ranges between 0 to 100. Symbol used for this term is P

outputFile='D1000I500P20.tsv'   #Specify the file name. 'D' represents the database size, 'I' represents the total number of items and 'P' represents the probability of occurrence of an item in a database
```
### Step 3: Creating the synthetic dataset
```Python
data = dbGenerator.generateTransactionalDatabase(totalNumberOfTransactions, totalNumberOfItems, probabilityOfOccurrenceOfAnItem, outputFile)
```

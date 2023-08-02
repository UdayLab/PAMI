# Mining Weighted Frequent Patterns in Uncertain Transactional Databases

## What is weighted frequent pattern mining?

weighted Frequent pattern mining aims to discover all interesting patterns in a transactional database that have **support** no less than the user-specified **minimum support** (**minSup**) constraint and **weight** no less than the user-specified **minimum weight** (**_minWeight_**).  The **minSup** controls the minimum number of transactions that a pattern must appear in a database. The **minWeight** controls the minimum weight of item. <br>

## What is the uncertain transactional database?

A transactional database is a collection of transactions, where each transaction contains a transaction-identifier and a set of items with ites repective uncertain value. <br> A hypothetical transactional database containing the items **_A, B, C, D, E, and F** as shown below

| tid | Transactions                  |
|-----|-------------------------------|
| 1   | B(0.5) C(0.45) F(1.0)         |
| 2   | A(0.7) B(0.82) D(0.3) F(0.75) |
| 3   | C(0.9) D(1.0) E(0.7)          | 
| 4   | A(0.48) B(0.8) C(0.6) D(1.0)  |
| 5   | B(0.7) D(0.3) E(1.0)          |
| 6   | B(0.65) C(1.0) D(0.8)         |
| 7   | C(0.9) D(0.5) F(1.0)          | 
| 8   | A(0.4) E(0.4)                 |
| 9   | A(0.8) B(1.0) D(0.8) F(0.7)   |
| 10  | B(0.4) C(0.9) D(1.0)          |

__Note:__  Duplicate items must not exist in a transaction.

## What is acceptable format of a transactional databases in PAMI

Each row in a transactional database must contain only items. The frequent pattern mining algorithms in PAMI implicitly assume the row number of a transaction as its transactional-identifier to reduce storage and processing costs. A sample transactional database, say [sample.txt](sample.txt), is provided below.

B(0.5) C(0.45) F(1.0)   <br>
A(0.7) B(0.82) D(0.3) F(0.75)  <br>
C(0.9) D(1.0) E(0.7)   <br>
A(0.48) B(0.8) C(0.6) D(1.0)   <br>
B(0.7) D(0.3) E(1.0)   <br>
B(0.65) C(1.0) D(0.8)  <br>
C(0.9) D(0.5) F(1.0)   <br>
A(0.4) E(0.4)    <br>
A(0.8) B(1.0) D(0.8) F(0.7)  <br>
B(0.4) C(0.9) D(1.0)   <br>

## What is the Weighted database?

A weight database is a collection of items with their weights. <br> 
A hypothetical weight database, say  [HEWIWeightSample.txt](HEWIWeightSample.txt), containing the items **_A, B, C, D, E and F_** as shown below

A 0.40  <br>
B 0.70  <br>
C 1.00  <br>
D 0.55  <br>
E 0.85  <br>
F 0.30  <br>

## Understanding the statisctics of database

To understand about the database. The below code will give the detail about the transactional database.
* Total number of transactions (Database size)
* Total number of unique items in database
* Minimum lenth of transaction that existed in database
* Average length of all transactions that exists in database
* Maximum length of transaction that existed in database
* Standard deviation of transaction length
* Variance in transaction length
* Sparsity of database

The below sample code prints the statistical details of a database.


```python
import PAMI.extras.dbStats.transactionalDatabaseStats as stats 
obj = stats.transactionalDatabaseStats('sample.txt', ' ') 
obj.run() 
obj.printStats() 
```

The input parameters to a frequent pattern mining algorithm are: 
* __Transactional database__  <br> Acceptable formats:
> * String : E.g., 'transactionalDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'Transactions'

* __minSup__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __minWeight__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __seperator__ <br> default seperator is '\t' (tab space)

## How to store the output of a uncertain weighted frequent pattern mining algorithm?
The patterns discovered by a correlated pattern mining algorithm can be saved into a file or a data frame.

## How to run the weighted frequent pattern mining algorithms in a terminal?

* Download the PAMI source code from github.
* Unzip the PAMI source code folder and enter into weighted uncertain frequent pattern folder.
* Enter into weightedUncertainFrequentPattern folder
* Enter into a specific folder and execute the  following command on terminal.

__syntax:__ python3 algorithmName.py `<path to the input file>` `<path to the output file>` `<path to the weight file>` `<minSup>` `<minWeight>` `<seperator>`

## Sample command to execute the WUFIM code in weightedUncertainFrequentPattern folder

__Example:__ python3 `WUFIM.py` `inputFile.txt` `outputFile.txt` `weightSample.txt` `3` `2` `' '`

## How to execute a weighted frequent pattern mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
import PAMI.weightedUncertainFrequentPattern.basic.WUFIM as alg 

iFile = 'sample.txt'  #specify the input transactional database <br>
wFile = 'HEWIWeightSample.txt'  #specify the input transactional database <br>
minSup = 1.4  #specify the minSupvalue <br>
minWeight = 1.5    #specify the minWeight value <br>
seperator = ' ' #specify the seperator. Default seperator is tab space. <br>
oFile = 'weightedPatterns.txt'   #specify the output file name<br>

obj = alg.WUFIM(iFile, wFile, minSup, minWeight, seperator) #initialize the algorithm <br>
obj.startMine()                       #start the mining process <br>
obj.savePatterns(oFile)               #store the patterns in file <br>
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe <br>
obj.printStats()                      #Print the statistics of mining process
```

The weightedPatterns.txt file contains the following patterns (*format:* pattern:support): !cat weightedPatterns.txt


```terminal
!cat weightedPatterns.txt
```

     E:2.1 
     C:4.75 
     C B:2.525 
     C B D:2.3 
     C D:3.65 
     B:4.870000000000001 
     B D:2.976 
     D:5.699999999999999 


The dataframe containing the patterns is shown below:


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Patterns</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(E,)</td>
      <td>2.100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(C,)</td>
      <td>4.750</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(C, B)</td>
      <td>2.525</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(C, B, D)</td>
      <td>2.300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(C, D)</td>
      <td>3.650</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(B,)</td>
      <td>4.870</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(B, D)</td>
      <td>2.976</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(D,)</td>
      <td>5.700</td>
    </tr>
  </tbody>
</table>
</div>



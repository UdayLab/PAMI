# Mining Frequent Patterns in Transactional Databases

## 1. What is frequent pattern mining?
Frequent pattern mining aims to discover all interesting patterns in a transactional database that have **support** no less than the user-specified **minimum support** (**minSup**) constraint.  The **minSup** controls the minimum number of transactions in which a pattern must appear in a database. <br>

Reference: Rakesh Agrawal, Tomasz Imieliński, and Arun Swami. 1993. Mining association rules between sets of items in large databases. In Proceedings of the 1993 ACM SIGMOD international conference on Management of data (SIGMOD '93). Association for Computing Machinery, New York, NY, USA, 207–216. [link](https://doi.org/10.1145/170035.170072)

## 2. What is a transactional database?
A transactional database is an unordered collection of transactions. A transaction represents a pair constituting of transaction-identifier and a set of items. <br> A hypothetical transactional database containing the items **_a, b, c, d, e, f, and g_** is shown below

|tid| Transactions|
| --- | --- |
| 1 | a b c g |
| 2 | b c d e |
| 3 | a b c d | 
| 4 | a c d f |
| 5 | a b c d g |
| 6 | c d e f |
| 7 | a b c d |
| 8 | a e f | 
| 9 | a b c d |
| 10 | b c d e |

__Note:__  Duplicate items must not exist within a transaction.

## 3. What is the acceptable format of a transactional database in PAMI?
Each row in a transactional database must contain only items. The frequent pattern mining algorithms in PAMI implicitly assume the row number of a transaction as its transactional-identifier to reduce storage and processing costs. <br>

A sample transactional database, say [sampleTransactionalDatabase.txt](sampleTransactionalDatabase.txt), is provided below.

a b c g <br>
b c d e <br>
a b c d <br>
a c d f <br>
a b c d g <br>
c d e f <br>
a b c d <br>
a e f <br>
a b c d <br>
b c d e <br>

## 4. What is the need for understanding the statistics of a transactional database?
The performance of a pattern mining algorithm primarily depends on the satistical nature of a database. Thus, it is important to know the following details of a database: 
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
import PAMI.extras.dbStats.TransactionalDatabase as stats 
obj = stats.TransactionalDatabase('sampleTransactionalDatabase.txt', ' ') 
obj.run() 
obj.printStats() 
```

## 5. What are the input parameters to a frequent pattern mining algorithm?
The input parameters to a frequent pattern mining algorithm are: 
* __Transactional database__  <br> Acceptable formats:
> * String : E.g., 'transactionalDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'Transactions'

* __minSup__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]
* __seperator__ <br> default seperator is '\t' (tab space)

## 6. How to store the output of a frequent pattern mining algorithm?
The patterns discovered by a frequent pattern mining algorithm can be saved into a file or a data frame.

## 7. How to run the frequent pattern mining algorithms in a terminal?
* Download the PAMI source code from [Github](https://github.com/udayRage/PAMI/archive/refs/heads/main.zip).
* Unzip the PAMI source code folder.
* Enter into frequentPattern folder.

```console
foo@bar: cd PAMI/frequentPattern/basic
```
* Execute the python program on ther terminal using the following syntax:

```console 
foo@bar:python3 algorithmName.py inputFile outputFile minSup seperator
```

__Example:__ python3 `Apriori.py` `inputFile.txt` `outputFile.txt` `3` `' '`

## 8. How to execute a frequent pattern mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
import PAMI.frequentPattern.basic.Apriori as alg 

iFile = 'sampleTransactionalDatabase.txt'  #specify the input transactional database 
minSup = 5                      #specify the minSup value 
seperator = ' '                  #specify the seperator. Default seperator is tab space. 
oFile = 'frequentPatterns.txt'   #specify the output file name

obj = alg.Apriori(iFile, minSup, seperator) #initialize the algorithm 
obj.startMine()                       #start the mining process
obj.save(oFile)               #store the patterns in file 
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe 
obj.printResults()                      #Print the stats of mining process
```

    Frequent patterns were generated successfully using Apriori algorithm 
    Total number of Frequent Patterns: 13
    Total Memory in USS: 81133568
    Total Memory in RSS 119091200
    Total ExecutionTime in ms: 0.00026297569274902344



```terminal
!cat frequentPatterns.txt
#format: frequentPattern:support
```

    a:7 
    b:7 
    c:9 
    d:8 
    b	a:5 
    c	a:6 
    c	b:7 
    b	d:6 
    c	d:8 
    d	a:5 
    c	b	a:5 
    c	b	d:6 
    c	d	a:5 


The dataframe containing the patterns is shown below:


```python
df #The dataframe containing the patterns is shown below. In each pattern, items were seperated from each other with a tab space (or \t). 
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
      <td>a</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b a</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c a</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>c b</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>b d</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>c d</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>d a</td>
      <td>5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>c b a</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>c b d</td>
      <td>6</td>
    </tr>
    <tr>
      <th>12</th>
      <td>c d a</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



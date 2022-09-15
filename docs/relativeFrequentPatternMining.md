# Mining Frequent Patterns with Relative Support in Transactional Databases

## 1. What is frequent pattern mining with relative support?

Frequent pattern mining aims to discover all interesting patterns in a transactional database that satisfy the user-specified **minimum support** (*minSup*) constraint. The *minSup* controls the minimum number of transactions that a pattern must cover in a database. Since only a single *minSup* is employed for the entire database, this technique implicitly assumes that all items in a database have uniform frequencies or similar occurrence behavior. However, this is seldom not the case in many real-world applications. In many applications, some items appear frequently in the data, while others occur rarely. If the frequencies of the items in a database vary a great deal, then finding frequent patterns with a single *minSup* leads to the following two problems:

- If *minSup* is set too high, we miss the patterns containing rare items. It is because rare items fail to satisfy the increased *minSup* value.
- To find patterns containing both frequent and rare items, the *minSup* has to be set very low. However, this may cause combinatorial explosion as frequent items may combine with one another in all possible ways.

This dillema is known as the ''rare item problem.''

When confronted with the above problem in the real-world applications, researchers tried to tackle it by finding frequent patterns using other measures, such as **relative support**. According to this extended model of frequent pattern, a pattern is said to be frequent if its **support** no less than the user-specified **minimum support** (**minSup**) constraint and **relative support** no less than the user-specified **minimum relative support** (**minRSup**).  The **minSup** controls the minimum number of transactions that a pattern must appear in a database. **minRSup** is the conditional minimum support calculated as ratio of pattern support to the minimum support of all items in pattern. <br>

References : 

- H. Yun, D. Ha, B. Hwang, and K. H. Ryu. Mining association rules on significant rare data using relative support. J. Syst. Softw., 67:181--191, September 2003. https://dl.acm.org/doi/10.1016/S0164-1212%2802%2900128-0

- R. Uday Kiran and Masaru Kitsuregawa. 2012. Towards efficient discovery of frequent patterns with relative support. In Proceedings of the 18th International Conference on Management of Data (COMAD '12). Computer Society of India, Mumbai, Maharashtra, IND, 92–99. https://dl.acm.org/doi/10.5555/2694443.2694460

## 2. What is a transactional database?

A transactional database is a collection of transactions, where each transaction contains a transaction-identifier and a set of items. <br> A hypothetical transactional database containing the items **_a, b, c, d, e, f, and g_** as shown below

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

__Note:__  Duplicate items must not exist in a transaction.

## 3. What is acceptable format of a transactional database in PAMI?

Each row in a transactional database must contain only items. The frequent pattern mining algorithms in PAMI implicitly assume the row number of a transaction as its transactional-identifier to reduce storage and processing costs. A sample transactional database, say sampleInputFile.txt, is provided below.

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
import PAMI.extras.dbStats.transactionalDatabaseStats as stats 
obj = stats.transactionalDatabaseStats('sampleTransactionalDatabase.txt', ' ') 
obj.run() 
obj.printStats() 
```

## 5. What are the input parameters to a relative frequent pattern mining algorithm?

Algorithms to mine the frequent patterns with relative support requires transactional database and minSup (specified by user).
* Transactional database can be provided in following formats:
> * String : E.g., 'transactionalDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame. Please note that dataframe must contain the header titled 'Transactions'
* __minSup__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]
* __minRelativeSupport__  <br> specified in 
> * [0, 1]
* __seperator__ <br> default seperator is '\t' (tab space)

## 6. How to store the output of a relative frequent pattern mining algorithm?
The patterns discovered by a relative frequent pattern mining algorithm can be saved into a file or a data frame.

## 7. How to run the relative frequent pattern mining algorithms in a terminal?

* Download the PAMI source code from [Github](https://github.com/udayRage/PAMI/archive/refs/heads/main.zip).
* Unzip the PAMI source code folder.
* Enter into relatedFrequentPattern folder.

```console
foo@bar: cd PAMI/relativeFrequentPattern/basic
```

* Execute the python program on ther terminal using the following syntax:

```console
foo@bar: python3 algorithmName.py inputFile outputFile minSup minRatio seperator
```


__Example:__  python3 `RSFPGrowth.py` `inputFile.txt` `outputFile.txt` `4` `0.7` `' '`

## 8. How to execute a frequent pattern mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
import PAMI.relativeFrequentPatterns.basic.RSFPGrowth as alg 

iFile = 'sampleTransactionalDatabase.txt'  #specify the input transactional database 
minSup = 5                     #specify the minSupvalue 
minRS = 0.6                 #specify the minimum relative support  
seperator = ' '                 #specify the seperator. Default seperator is tab space. 
oFile = 'frequentPatterns.txt'   #specify the output file name

obj = alg.RSFPGrowth(iFile, minSup, minRatio, seperator) #initialize the algorithm 
obj.startMine()                       #start the mining process 
obj.save(oFile)               #store the patterns in file 
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe 
obj.printResults()                      #Print the results of mining process
```

    Relative support frequent patterns were generated successfully using RSFPGrowth algorithm
    Total number of Relative Frequent Patterns: 15
    Total Memory in USS: 98238464
    Total Memory in RSS 135536640
    Total ExecutionTime in ms: 0.0007281303405761719



```python
!cat frequentPatterns.txt
#format: relativeFrequentPattern:support:relativeRatio
```

    d: 8 : 1.0 
    d	c: 16 : 2.0 
    c: 9 : 1.0 
    b: 7 : 1.0 
    b	d: 12 : 1.7142857142857142 
    b	d	c: 12 : 1.7142857142857142 
    b	c: 14 : 2.0 
    b	a: 10 : 1.4285714285714286 
    b	a	c: 8 : 1.1428571428571428 
    b	a	d: 8 : 1.1428571428571428 
    b	a	c	d: 8 : 1.1428571428571428 
    a: 7 : 1.0 
    a	c: 10 : 1.4285714285714286 
    a	d: 10 : 1.4285714285714286 
    a	c	d: 10 : 1.4285714285714286 


The dataframe contains the following information:


```python
df
#The dataframe containing the patterns is shown below. In each pattern, items were seperated from each other with a tab space (or \t). 
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
      <td>d</td>
      <td>8 : 1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d c</td>
      <td>16 : 2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>9 : 1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>7 : 1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b d</td>
      <td>12 : 1.7142857142857142</td>
    </tr>
    <tr>
      <th>5</th>
      <td>b d c</td>
      <td>12 : 1.7142857142857142</td>
    </tr>
    <tr>
      <th>6</th>
      <td>b c</td>
      <td>14 : 2.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>b a</td>
      <td>10 : 1.4285714285714286</td>
    </tr>
    <tr>
      <th>8</th>
      <td>b a c</td>
      <td>8 : 1.1428571428571428</td>
    </tr>
    <tr>
      <th>9</th>
      <td>b a d</td>
      <td>8 : 1.1428571428571428</td>
    </tr>
    <tr>
      <th>10</th>
      <td>b a c d</td>
      <td>8 : 1.1428571428571428</td>
    </tr>
    <tr>
      <th>11</th>
      <td>a</td>
      <td>7 : 1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>a c</td>
      <td>10 : 1.4285714285714286</td>
    </tr>
    <tr>
      <th>13</th>
      <td>a d</td>
      <td>10 : 1.4285714285714286</td>
    </tr>
    <tr>
      <th>14</th>
      <td>a c d</td>
      <td>10 : 1.4285714285714286</td>
    </tr>
  </tbody>
</table>
</div>



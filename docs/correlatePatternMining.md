# Mining Correlated Patterns in Transactional Databases

## 1. What is correlated pattern mining?

Frequent pattern mining aims to discover all interesting patterns in a transactional database that satisfy the user-specified **minimum support** (*minSup*) constraint. The *minSup* controls the minimum number of transactions that a pattern must cover in a database. Since only a single *minSup* is employed for the entire database, this technique implicitly assumes that all items in a database have uniform frequencies or similar occurrence behavior. However, this is seldom not the case in many real-world applications. In many applications, some items appear frequently in the data, while others occur rarely. If the frequencies of the items in a database vary a great deal, then finding frequent patterns with a single *minSup* leads to the following two problems:

- If *minSup* is set too high, we miss the patterns containing rare items. It is because rare items fail to satisfy the increased *minSup* value.
- To find patterns containing both frequent and rare items, the *minSup* has to be set very low. However, this may cause combinatorial explosion as frequent items may combine with one another in all possible ways.

This dillema is known as the ''rare item problem.''

When confronted with the above problem in the real-world applications, researchers tried to tackle it by finding correlated patterns in a database. Several alternative measures have been described in the literature to find correlated patterns. Each measure has a selection bias that justifies the significance of one pattern over another. Consequently, there exists no universally accepted best measure to find correlated patterns. However, finding correlated patterns using *all-confidence* measure has gained popularity as it satisfies both *null-invariant* and *anti-monotonic properities*.  In this context, we have developed correlated pattern mining algorithms using *all-confidence measure*.

According to the *all-confidence* based correlated pattern mining model, a pattern is said to be **correlated** if it satisfies both *minimum Support* and *minimum all-confidence* constraints. 

**References:**  

- E. R. Omiecinski, "Alternative interest measures for mining associations in databases," in IEEE Transactions on Knowledge and Data Engineering, vol. 15, no. 1, pp. 57-69, Jan.-Feb. 2003, doi: 10.1109/TKDE.2003.1161582. [Link](https://ieeexplore.ieee.org/document/1161582)

- Young-Koo Lee, Won-Young Kim, Y. Dora Cai, Jiawei Han: CoMine: Efficient Mining of Correlated Patterns. 581-584. [link](https://ieeexplore.ieee.org/document/1250982)


## 2. What is a transactional database?
A transactional database is a collection of transactions, where each transaction contains a transaction-identifier and a set of items. <br> A hypothetical transactional database containing the items **_a, b, c, d, e, f, and g_** is shown below

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
import PAMI.extras.dbStats.transactionalDatabaseStats as stats 
obj = stats.transactionalDatabaseStats('sampleTransactionalDatabase.txt', ' ') 
obj.run() 
obj.printStats() 
```

## 5. What are the input parameters to a correlated pattern mining algorithm?

The input parameters to a correlated pattern mining algorithm are: 
* __Transactional database__  <br> Acceptable formats:
> * String : E.g., 'transactionalDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'Transactions'

* __minSup__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __minAllConf__  <br> specified in 
> * [0, 1]

* __seperator__ <br> default seperator is '\t' (tab space)

## 6. How to store the output of a correlated pattern mining algorithm?
The patterns discovered by a correlated pattern mining algorithm can be saved into a file or a data frame.

## 7. How to run a correlated pattern mining algorithm in a terminal?
* Download the PAMI source code from [Github](https://github.com/udayRage/PAMI/archive/refs/heads/main.zip).
* Unzip the PAMI source code folder.
* Enter into correlatedPattern folder.
```console
foo@bar: cd PAMI/correlatedPattern/basic
```
* Execute the python program on the terminal using the following syntax:
```console
foo@bar: python3 algorithmName.py inputFile outputFile minSup minAllConf seperator
```

__Example:__ python3 `CPGrowth.py` `inputFile.txt` `outputFile.txt` `3` `0.4` `' '`

## 8. How to execute a correlated pattern mining algorithm in a Jupyter Notebook?
- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes

```python
import PAMI.correlatedPattern.basic.CPGrowth as alg

iFile = 'sampleTransactionalDatabase.txt'  # specify the input transactional database
minSup = 4  # specify the minSupvalue
minAllConf = 0.7  # specify the minAllConf value
seperator = ' '  # specify the seperator. Default seperator is tab space.
oFile = 'correlatedPattern.txt'  # specify the output file name<

obj = alg.CPGrowth(iFile, minSup, minAllConf, seperator)  # initialize the algorithm
obj.startMine()  # start the mining process
obj.save(oFile)  # store the patterns in file
df = obj.getPatternsAsDataFrame()  # Get the patterns discovered into a dataframe
obj.printResults()     
```

    Correlated Frequent patterns were generated successfully using CorrelatedPatternGrowth algorithm
    Total number of Correlated Patterns: 9
    Total Memory in USS: 81182720
    Total Memory in RSS 119152640
    Total ExecutionTime in ms: 0.0003771781921386719



```python
!cat correlatedPatterns.txt 
#format: correlatedPattern:support:all-confidence
```

    e:4:1.0 
    d:8:1.0 
    d	c:8:0.8888888888888888 
    c:9:1.0 
    b:7:1.0 
    b	d:6:0.75 
    b	c:7:0.7777777777777778 
    b	a:5:0.7142857142857143 
    a:7:1.0 


The dataframe contains the following information:


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
      <th>Confidence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>e</td>
      <td>4</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d</td>
      <td>8</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d c</td>
      <td>8</td>
      <td>0.888889</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c</td>
      <td>9</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>7</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>b d</td>
      <td>6</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>b c</td>
      <td>7</td>
      <td>0.777778</td>
    </tr>
    <tr>
      <th>7</th>
      <td>b a</td>
      <td>5</td>
      <td>0.714286</td>
    </tr>
    <tr>
      <th>8</th>
      <td>a</td>
      <td>7</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



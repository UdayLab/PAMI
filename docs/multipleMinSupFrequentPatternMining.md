# Mining Frequent Patterns With Multiple Minimum Supports in Transactional Databases

## 1. What is frequent pattern mining with multiple minimum support?

Frequent pattern mining aims to discover all interesting patterns in a transactional database that satisfy the user-specified **minimum support** (*minSup*) constraint. The *minSup* controls the minimum number of transactions that a pattern must cover in a database. Since only a single *minSup* is employed for the entire database, this technique implicitly assumes that all items in a database have uniform frequencies or similar occurrence behavior. However, this is seldom not the case in many real-world applications. In many applications, some items appear frequently in the data, while others occur rarely. If the frequencies of the items in a database vary a great deal, then finding frequent patterns with a single *minSup* leads to the following two problems:

- If *minSup* is set too high, we miss the patterns containing rare items. It is because rare items fail to satisfy the increased *minSup* value.
- To find patterns containing both frequent and rare items, the *minSup* has to be set very low. However, this may cause combinatorial explosion as frequent items may combine with one another in all possible ways.

This dillema is known as the ''rare item problem.''

When confronted with the above problem in the real-world applications, researchers tried to tackle it by finding the frequent patterns using multiple minimum support values. In this extended model, each item in the database is specified with a *minSup*-like constraint known as *minimum item support*, and the *minSup* of a pattern is expressed with the lowest minimum support of its items. 


References: 
- Bing Liu, Wynne Hsu, and Yiming Ma. 1999. Mining association rules with multiple minimum supports. In Proceedings of the fifth ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '99). Association for Computing Machinery, New York, NY, USA, 337–341. [Link](https://doi.org/10.1145/312129.312274)

- R. Uday Kiran, P. Krishna Reddy: Novel techniques to reduce search space in multiple minimum supports-based frequent pattern mining algorithms. EDBT 2011: 11-20. [Link](https://dl.acm.org/doi/10.1145/1951365.1951370)


## 2. What is a transactional database?

A transactional database is a collection of transactions, where each transaction contains a transaction-identifier and a set of items. <br> A hypothetical transactional database containing the items **_a, b, c, d, e, f, and g_** as shown below.

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

## 3. What is acceptable format of a transactional databases in PAMI?

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

## 4. What is the need for understanding the statisctics of a transactional database?

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

## 5. What is a multiple minimum support file?

A multiple minimum support file contains a set of items and its respective minimum support value. <br> A hypothetical multiple minSup database containing the items **_a, b, c, d, e, f, and g_** as shown below.

|Item| MIS |
| --- | --- |
| a | 4 |
| b | 4 |
| c | 3 | 
| d | 4 |
| e | 4 |
| f | 5 |
| g | 5 |


In many real-world applications, it is often difficult for the users to specify minimum item supports for every item in a database. Different methods have been described in the literature to derive the items' *minimum item support* values. The procedures to specify the items' *MIS* values is provided at [derivingItemsMISValues](derivingItemsMISValues.html).

## 6. What is the acceptable format of multiple minimum support in PAMI?

a 4  <br>
b 4  <br>
c 3  <br>
d 4  <br>
e 4  <br>
f 5  <br>
g 5  <br>

## 7. What are the input parameters to a multiple minimum support based frequent pattern mining algorithm?

The input parameters to a frequent pattern mining algorithm are: 
* __Transactional database__  <br> Acceptable formats:
> * String : E.g., 'transactionalDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'Transactions'

* __multiple minSup value database__  <br> Acceptable formats:
> * String : E.g., 'minSupDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'item' and 'MIS'

* __seperator__ <br> default seperator is '\t' (tab space)

## 8. How to store the output of a frequent pattern mining with multiple minimum support algorithm?
The patterns discovered by a frequent pattern mining with multiple minimum support algorithm can be saved into a file or a data frame.

* Download the PAMI source code from github.
* Unzip the PAMI source code folder.
* Enter into multipleminimumSupportbasedFrequentPattern folder.
```console
foo@bar: cd PAMI/multipleminimumSupportbasedFrequentPattern/basic
```
* Execute the python program on the terminal using the following syntax:
```console
foo@bar: python3 algorithmName.py inputFile outputFile multipleMinSupFile.txt seperator
```




__Example:__ python3 `CFPGrowth.py` `inputFile.txt` `outputFile.txt` `MIS.txt` `' '`

## 9. How to execute a frequent pattern mining with multiple minimum supports algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
import PAMI.multipleMinimumSupportBasedFrequentPattern.basic.CFPGrowth as alg 

iFile = 'sampleTransactionalDatabase.txt'     #specify the input transactional database 
mFile = 'MIS.txt'                 #specify the multiple minsup file
seperator = ' '                  #specify the seperator. Default seperator is tab space. 
oFile = 'frequentPatterns.txt'   #specify the output file name

obj = alg.CFPGrowth(iFile, mFile, seperator) #initialize the algorithm 
obj.startMine()                       #start the mining process 
obj.save(oFile)               #store the patterns in file 
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe 
obj.printResults()                      #Print the statistics of mining process
```

    Frequent patterns were generated successfully using CFPGrowth algorithm
    Total number of  Frequent Patterns: 10
    Total Memory in USS: 100147200
    Total Memory in RSS 138452992
    Total ExecutionTime in ms: 0.0017397403717041016



```python
!cat frequentPatterns.txt
#format: frequentPattern:support
```

    e:4 
    e	d	c:3 
    e	c:3 
    b:5 
    b	d:6 
    b	d	c:6 
    b	c:7 
    d:8 
    d	c:8 
    c:8 


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
      <td>e</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>e d c</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e c</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b d</td>
      <td>6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>b d c</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>b c</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>d</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>d c</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>c</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



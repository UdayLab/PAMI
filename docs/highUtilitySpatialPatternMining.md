# Mining High-Utility Spatial Patterns in Utility Databases

## 1. What is High-Utility Spatial pattern mining?

High utility pattern mining aims to discover all the patterns with utility of pattern is no less than user-specified **_minimum utility_** threshold **_minutil_** and no distance between any two of its items should be no greater than user-specified **_maximum distance_**.

Reference:  R. Uday Kiran, Koji Zettsu, Masashi Toyoda, Philippe Fournier-Viger, P. Krishna Reddy, and Masaru Kitsuregawa. 2019. Discovering Spatial High Utility Itemsets in Spatiotemporal Databases. In Proceedings of the 31st International Conference on Scientific and Statistical Database Management (SSDBM '19). Association for Computing Machinery, New York, NY, USA, 49–60. https://doi.org/10.1145/3335783.3335789

## 2. What is a utility database?

A utility database is a collection of transaction, where each transaction contains a set of items and a positive integer called **_internal utility_** respectively. And each unique item in database is also associated with another positive number called **_external utility_** for each transaction. <br>
A hypothetical utility database with items **_a, b, c, d, e, f and g_** and its **_internal utility_** is shown below at right side and items with its **_external utilities_** for each transaction is presented at left side.

| Transactions                  | external utilities |
|-------------------------------|--------------------|
| (a,2) (b,3) (c,1) (g,1)       | 5 4 3 2            |
| (b,3) (c,2) (d,3) (e,2)       | 5 2 9 3            |
| (a,2) (b,1) (c,3) (d,4)       | 2 3 5 6            |
| (a,3) (c,2) (d,1) (f,2)       | 1 3 4 6            |
| (a,3) (b,1) (c,2) (d,1) (g,2) | 2 5 3 6 1          |
| (c,2) (d,2) (e,3) (f,1)       | 2 3 4 5            |
| (a,2) (b,1) (c,1) (d,2)       | 5 4 3 2            |
| (a,1) (e,2) (f,2)             | 4 8 3              |
| (a,2) (b,2) (c,4) (d,2)       | 7 4 9 8            |
| (b,3) (c,2) (d,2) (e,2)       | 5 9 10 24          |

__Note:__  Duplicate items must not exist in a transaction.

## 3. What is the acceptable format of a utility database in PAMI?

Each row in a utility database must contain only items, total sum of utilties and utility values. A sample transactional database, say sampleInputFile.txt, is provided below. <br>

A sample utility database, say [sampleUtilitySpatial.txt](sampleUtilitySpatial.txt), is provided below.

a b c g:7:2 3 1 1:5 4 3 2   <br>
b c d e:10:3 2 3 2:5 2 9 3  <br>
a b c d:10:2 1 3 4:2 3 5 6  <br>
a c d f:7:3 2 1 2:1 3 4 6   <br>
a b c d g:9:3 1 2 1 2:2 5 3 6 1   <br>
c d e f:8:2 2 3 1:2 3 4 5   <br>
a b c d:6:2 1 1 2:5 4 3 2   <br>
a e f:5:1 2 2:4 8 3         <br>
a b c d:10:2 2 4 2:7 4 9 8  <br>
b c d e:9:3 2 2 2:5 9 10 24  <br>

## 4. What is a neighbourhood database? 

A neighborhood database contains items and their neighbors. An item *x* is said to be a neighbor of *y* if the distance between *x* and *y* is no more than the user-specified *maximum distance* threshold value.<br>
A hypothetical spatial database containing items **_a, b, c, d, e, f and g_** and neighbours respectively is shown below.

| Items | neighbours |
| --- | --- |
| a | b, c, d |
| b | a, e, g |
| c | a, d | 
| d | a, c |
| e | b, f |
| f | e, g |
| g | b, f |

The methodology to create a neighborhood database file from a given geo-referenced database has been described in the manual [creatingNeighborhoodFile.pdf](creatingNeighborhoodFile.pdf)

## 5. What is the acceptable format of a neighborhood database?

The format of the neighborhood database is similar to that of a transactional database. That is, each transaction must contain a set of items. In a transaction, the first item represents the key item, while the remaining items represent the neighbors of the first item. 

A sample neighborhood file, say [sampleNeighbourFile.txt](sampleNeighbourFile.txt), is provided below:

a b c d <br>
b a e g<br>
c a d<br>
d a c<br>
e b f<br>
f e g<br>
g b f <br>

## 6. What is the need for understanding the statistics of a utility database?

The performace of a pattern mining algorithm primarily depends on the satistical nature of a database. Thus, it is important to know the following details of a database: 
* Total number of transactions (Database size)
* Total number of unique items in database
* Minimum lenth of transaction that existed in database
* Average length of all transactions that exists in database
* Maximum length of transaction that existed in database
* Minimum utility value exists in database
* Average utility exists in database
* Maximum utility exists in database
* Standard deviation of transaction length
* Variance in transaction length
* Sparsity of database

The below sample code prints the statistical details of a database.

```python
import PAMI.extras.dbStats.UtilityDatabase as stats

obj = stats.UtilityDatabase('sampleInputFile.txt', ' ')
obj.run()
obj.printStats() 
```

## 7. What are the input parameters to a high utility spatial pattern mining algorithm?

Algorithms to mine the high utility spatial patterns require utility database, neighborhood database, and a user-specified *minUtil* constraint. Please note that *maxDist* constraint has been used in prior to create a neighborhood database file.

* __Utility database__  <br> Acceptable formats:
> * String : E.g., 'utilityDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'Transactions', 'Utility' and 'TransactionUtility'

* __Spatial database__  <br> Acceptable formats:
> * String : E.g., 'spatialDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'item' and 'Neighbours'

* __minUtil__  <br> specified in 
> * __count__
* __seperator__ <br> default seperator is '\t' (tab space)

## 8.How to store the output of a frequent pattern mining algorithm?
The patterns discovered by a high utility spatial pattern mining algorithm can be saved into a file or a data frame.

## 9. How to execute a high utility spatial pattern mining algorithm in a terminal?

* Download the PAMI source code from [Github](https://github.com/udayRage/PAMI/archive/refs/heads/main.zip)..
* Unzip the PAMI source code folder.
* Enter into highUtilitySpatialPattern folder.
```console
foo@bar: cg PAMI/highUtilitySpatialPattern/basiv
```
* Execute the python program on theri terminal using the following syntax:
```console
foo@bar: python3 algorithmName.py inputFile outputFile neighbourFile minUtil seperator
```


__Example:__ python3 `HDSHUIM.py` `inputFile.txt` `outputFile.txt` `neighbourFile.txt` $20$ &nbsp; `' '`

## 10. How to execute a high utility spatial pattern mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
import PAMI.highUtilitySpatialPattern.basic.HDSHUIM as alg 

iFile = 'sampleUtilitySpatial.txt'  #specify the input utility database
nFile = 'sampleNeighbourFile.txt'      #specify the neighbour file of database 
minUtil = 20                        #specify the minUtil value 
seperator = ' ' #specify the seperator. Default seperator is tab space. 
oFile = 'utilityPatterns.txt'   #specify the output file name

obj = alg.HDSHUIM(iFile, nFile, minUtil, seperator) #initialize the algorithm
obj.startMine()                       #start the mining process
obj.save(oFile)               #store the patterns in file
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe
obj.printResults()                      #Print the stats of mining process
```

    7
    Total number of Spatial High Utility Patterns: 4
    Total Memory in USS: 114114560
    Total Memory in RSS 152379392
    Total ExecutionTime in seconds: 0.0016164779663085938



```python
!cat utilityPatterns.txt
#The format of the file is pattern:support
```

    a	d:22 
    a	d	c:34 
    a	c:27 
    d	c:35 



```python
df
# zthe dataframe containing the patterns is shown below.
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
      <td>a d</td>
      <td>22</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a d c</td>
      <td>34</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a c</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d c</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>



# Mining Partial Periodic Spatial Patterns in Temporal Databases

## 1. What is partial periodic spatial pattern mining?

Partial periodic spatial pattern mining aims to discover all interesting patterns in a geo-referenced temporal database that have **periodic support** no less than the user-specified **minimum periodic support** (**minPS**) constraint and the **distance** between two items is no less than **maximum distance** (**maxDist**). The **minPS** controls the minimum number of periodic occurrences of a pattern in a database.

## 2. What is the temporal database?

A temporal database is a collection of transactions at a particular timestamp, where each transaction contains a timestamp and a set of items. <br> A hypothetical temporal database containing the items **_a, b, c, d, e, f, and g_** as shown below

| TS  | Transactions |
|-----|--------------|
| 1   | a b c g      |
| 2   | b c d e      |
| 3   | a b c d      | 
| 4   | a c d f      |
| 5   | a b c d g    |
| 6   | c d e f      |
| 7   | a b c d      |
| 8   | a e f        | 
| 9   | a b c d      |
| 10  | b c d e      |

__Note:__  Duplicate items must not exist in a transaction.

## 3. What is the acceptable format of a temporal database in PAMI?

Each row in a temporal database must contain timestamp and items. A sample temporal database, say [sampleTemporalDatabase.txt](sampleTemporalDatabase.txt), is show below.

1 a b c g <br>
2 b c d e <br>
3 a b c d <br>
4 a c d f <br>
5 a b c d g <br>
6 c d e f <br>
7 a b c d <br>
8 a e f <br>
9 a b c d <br>
10 b c d e <br>

## 4. What is the neighborhood database? 

Neighborhood database contains the information regarding the items and their neighboring items.

| Items | neighbours |
| --- | --- |
| a | b, c, d |
| b | a, e, g |
| c | a, d | 
| d | a, c |
| e | b, f |
| f | e, g |
| g | b, f |

## 5. What is the need for understand the statisctics of database?

To understand about the database. The below code will give the detail about the transactional database.
* Total number of transactions (Database size)
* Total number of unique items in database
* Minimum lenth of transaction that existed in database
* Average length of all transactions that exists in database
* Maximum length of transaction that existed in database
* Minimum periodicity exists in database
* Average periodicity exists in database
* Maximum periodicity exists in database
* Standard deviation of transaction length
* Variance in transaction length
* Sparsity of database

The below sample code prints the statistical details of a database.


```python
import PAMI.extras.dbStats.temporalDatabaseStats as stats 
obj = stats.temporalDatabaseStats('sampleTemporalDatabase.txt', ' ') 
obj.run() 
obj.printStats()
```

    Database size : 10
    Number of items : 7
    Minimum Transaction Size : 3
    Average Transaction Size : 4.0
    Maximum Transaction Size : 5
    Minimum period : 1
    Average period : 1.0
    Maximum period : 1
    Standard Deviation Transaction Size : 0.4472135954999579
    Variance : 0.2222222222222222
    Sparsity : 0.42857142857142855


## 6. What are the input parameters?

The input parameters to a partial periodic spatial pattern mining algorithm are: 
* __Temporal database__  <br> Acceptable formats:
> * String : E.g., 'temporalDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'TS' and 'Transactions'

* __Neighbour database__  <br> Acceptable formats:
> * String : E.g., 'neighbourDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'Item' and 'neighbours'

* __minPS__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __maxIAT__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __seperator__ <br> default seperator is '\t' (tab space)

## 7. How to store the output of a partial periodic spatial pattern mining algorithm?
The patterns discovered by a partial periodic spatial pattern mining algorithm can be saved into a file or a data frame.

## 8. How to run the partial periodic spatial pattern mining algorithms in a terminal?

* Download the PAMI source code from github.
* Unzip the PAMI source code folder and enter into partial periodic spatial pattern folder.
* Enter into partialPeriodicSpatialPattern folder
* Enter into a specific folder of your choice and execute the  following command on terminal.

__syntax:__ python3 algorithmName.py `<path to the input file>` `<path to the output file>` `<path to the neighbour file>` `<minPS>` `<maxIAT>` `<seperator>`

__Example:__ python3 `STECLAT.py` `inputFile.txt` `outputFile.txt` `neighbourFile.txt` `3`  `4` `' '`

## 10. How to execute a partial periodic spatial pattern mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes

```python
import PAMI.georeferencedPartialPeriodicPattern.basic.STEclat as alg

iFile = 'sampleTemporalDatabase.txt'  # specify the input transactional database <br>
nFile = 'sampleNeighbourFile.txt'  # specify the input transactional database <br>
minPS = 5  # specify the minSupvalue <br>
maxIAT = 3  # specify the minSupvalue <br>
seperator = ' '  # specify the seperator. Default seperator is tab space. <br>
oFile = 'partialSpatialPatterns.txt'  # specify the output file name<br>

obj = alg.STEclat(iFile, nFile, minPS, maxIAT, seperator)  # initialize the algorithm <br>
obj.startMine()  # start the mining process <br>
obj.save(oFile)  # store the patterns in file <br>
df = obj.getPatternsAsDataFrame()  # Get the patterns discovered into a dataframe <br>
obj.printResults()  # Print the statistics of mining process
```

    Spatial Periodic Frequent patterns were generated successfully using SpatialEclat algorithm
    Total number of  Spatial Partial Periodic Patterns: 6
    Total Memory in USS: 129249280
    Total Memory in RSS 170835968
    Total ExecutionTime in ms: 0.0015385150909423828


The partialSpatialPatterns.txt file contains the following patterns (*format:* pattern:periodicSupport):!cat partialSpatialPatterns.txt


```terminal
!cat partialSpatialPatterns.txt
```

    c	d: 7 
    c	a: 5 
    c: 8 
    d: 7 
    a: 6 
    b: 6 


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
      <th>periodicSupport</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c\td\t</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>c\ta\t</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c\t</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d\t</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a\t</td>
      <td>6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>b\t</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



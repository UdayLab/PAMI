# Mining Geo Referenced Periodic-Frequent Patterns in Temporal Databases

## 1. What is geo referenced periodic-frequent pattern mining?

Geo Referenced Periodic-Frequent pattern mining aims to discover all interesting patterns in a temporal database that have **support** no less than the user-specified **minimum support** (**minSup**) constraint,  **periodicity** no greater than user-specified **maximum periodicity** (**maxPer**) constraint and **distance** between two items is no less than **maximum distance** (**maxDist**). The **minSup** controls the minimum number of transactions that a pattern must appear in a database and the **maxPer** controls the maximum time interval within which a pattern must reappear in the database. 

## 2. What is a temporal database?

A temporal database is a collection of transactions at a particular timestamp, where each transaction contains a timestamp and a set of items. <br> A hypothetical temporal database containing the items **_a, b, c, d, e, f, and g_** as shown below

|TS| Transactions|
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

## 3. Acceptable format of temporal databases in PAMI

Each row in a temporal database must contain timestamp and items.

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

## 4. What is the spatial database? 

Spatial database contain the spatial (neighbourhood) information of items. It contains the items and its nearset neighbours satisfying the **maxDist** constraint.

| Items | neighbours |
| --- | --- |
| a | b, c, d |
| b | a, e, g |
| c | a, d | 
| d | a, c |
| e | b, f |
| f | e, g |
| g | b, f |

## 5. Understanding the statisctics of database

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
import PAMI.extras.dbStats.TemporalDatabase as stats

obj = stats.TemporalDatabase('sampleInputFile.txt', ' ')
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


## 5. What are the input parameters?

The input parameters to a periodic frequent spatial pattern mining algorithm are: 
* __Temporal database__  <br> Acceptable formats:
> * String : E.g., 'temporalDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'TS' and 'Transactions'

* __Neighbour database__  <br> Acceptable formats:
> * String : E.g., 'NeighbourDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'TS' and 'Transactions'

* __minSup__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __maxPer__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __seperator__ <br> default seperator is '\t' (tab space)


## 6. How to store the output of a geo referenced periodic frequent pattern mining algorithm?
The patterns discovered by a geo referenced periodic frequent pattern mining algorithm can be saved into a file or a data frame.

## 7. How to run the geo referenced periodic frequent pattern mining algorithms in a terminal?

* Download the PAMI source code from github.
* Unzip the PAMI source code folder and enter into geo referenced periodic frequent pattern folder.
* Enter into geoReferencedPeriodicFrequentPattern folder
* Enter into a specific folder of your choice and execute the  following command on terminal.

__syntax:__ python3 algorithmName.py `<path to the input file>` `<path to the output file>` `<path to the neighbour file>` `<minSup>` `<maxPer>` `<seperator>`

__Example:__ python3 `GPFPMiner.py` `inputFile.txt` `outputFile.txt` `neighbourFile.txt` `3` `4` `' '`

## 8. How to implement the GPFPMiner algorithm by importing PAMI package

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
import PAMI.geoReferencedPeriodicFrequentPattern.basic.GPFPMiner as alg

iFile = 'sampleInputFile.txt'  #specify the input temporal database <br>
nFile = 'sampleNeighbourFile.txt'  #specify the input neighbour database <br>
minSup = 5  #specify the minSupvalue <br>
maxPer = 3  #specify the maxPer value <br>
seperator = ' ' #specify the seperator. Default seperator is tab space. <br>
oFile = 'Patterns.txt'   #specify the output file name<br>

obj = alg.GPFPMiner(iFile, nFile, minSup, maxPer, seperator) #initialize the algorithm <br>
obj.mine()                       #start the mining process <br>
obj.save(oFile)               #store the patterns in file <br>
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe <br>
obj.printResults()                      #Print the stats of mining process
```

    Spatial Periodic Frequent patterns were generated successfully using SpatialEclat algorithm
    Total number of Spatial Periodic-Frequent Patterns: 9
    Total Memory in USS: 115994624
    Total Memory in RSS 156839936
    Total ExecutionTime in seconds: 0.0010027885437011719


The Patterns.txt file contains the following patterns (*format:* pattern:support:periodicity):!cat Patterns.txt


```python
!cat Patterns.txt
```

    d	c	a	: 5: 3 
    c	d	: 8: 2 
    c	a	: 6: 2 
    c	: 9: 2 
    d	a	: 5: 3 
    d	: 8: 2 
    b	a	: 5: 2 
    b	: 7: 2 
    a	: 7: 2 


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
      <th>Period</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>d\tc\ta\t</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>c\td\t</td>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c\ta\t</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c\t</td>
      <td>9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>d\ta\t</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>d\t</td>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>b\ta\t</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>b\t</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>a\t</td>
      <td>7</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

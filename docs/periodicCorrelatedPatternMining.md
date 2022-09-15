# Mining periodic correlated patterns in a temporal database

## 1. What is periodic correlated pattern mining?

Periodic correlated pattern mining aims to discover all the interesting patterns in a temporal database that satisfy the user-specified  **minimum support (minSup)**, *minimum all confidence (minAllConf)**,   **maximum periodicity (maxPer)**, and **maximum period all-confidence (maxPerAllConf)**

Reference: Venkatesh, J.N., Uday Kiran, R., Krishna Reddy, P., Kitsuregawa, M. (2018). Discovering Periodic-Correlated Patterns in Temporal Databases. In: Hameurlain, A., Wagner, R., Hartmann, S., Ma, H. (eds) Transactions on Large-Scale Data- and Knowledge-Centered Systems XXXVIII. Lecture Notes in Computer Science(), vol 11250. Springer, Berlin, Heidelberg. [Link](https://doi.org/10.1007/978-3-662-58384-5_6)

## 2. What is a temporal database?

A temporal database is an unordered collection of transactions. A temporal represents a pair constituting of temporal-timestamp and a set of items. <br> A hypothetical temporal database containing the items **_a, b, c, d, e, f, and g_**  and its timestamp is shown below

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

__Note:__  Duplicate items must not exist within a transaction.

## 3. What is the acceptable format of a temporal database in PAMI?

Each row in a temporal database must contain timestamp and items.  A sample transactional database, say sampleInputFile.txt, is provided below.

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

## 4. What is the need for understand the statisctics of database?

The performance of a pattern mining algorithm primarily depends on the satistical nature of a database. Thus it is important to know the following details of a database: 
* Total number of transactions (Database size)
* Total number of unique items in database
* Minimum lenth of transaction that exists in database
* Average length of all transactions that exists in database
* Maximum length of transaction that exists in database
* Minimum periodicity that exists in database
* Average periodicity hat exists in database
* Maximum periodicity that exists in database
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


## 5. What are the input parameters?

The input parameters to a periodic frequent pattern mining algorithm are: 
* __Temporal database__  <br> Acceptable formats:
> * String : E.g., 'transactionalDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'TS' and 'Transactions'

* __minSup__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __minAllConf__  <br> specified in 
> * [0, 1]

* __maxPer__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]
* __maxPerAllConf__  <br> specified in 
> * [0, 1]

* __seperator__ <br> default seperator is '\t' (tab space)

## 6. How to store the output of a correlated periodic  pattern mining algorithm?
The patterns discovered by a periodic correlated pattern mining algorithm can be saved into a file or a data frame.

## 7. How to run the correlated periodic pattern mining algorithms in a terminal?

* Download the PAMI source code from github.
* Unzip the PAMI source code folder and enter into periodic correlated pattern folder.
* Enter into periodicCorrelatedPattern folder
* Enter into specific folder execute the  following command on terminal.

__syntax:__ python3 algorithmName.py `<path to the input file>` `<path to the output file>` `<minSup>` `<minAllConf>` `<maxPer>` `<maxPerAllConf>`  `<seperator>`

__Example:__ python3 `EPCPGrowth` `inputFile.txt` `outputFile.txt` `4` `0.5` `3` `0.4` `' '`

## 8. How to execute a periodic correlated pattern mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
import PAMI.periodicCorrelatedPattern.EPCPGrowth as alg 

iFile = 'sampleInputFile.txt'  #specify the input temporal database 
minSup = 4                     #specify the minSup value 
minAllConf = 0.6               #specify the minAllConf value 
maxPer = 4                     #specify the maxPer value <br>
maxPerAllConf = 1.5            #specify the maxPerAllConf Value <br>
seperator = ' '                 #specify the seperator. Default seperator is tab space. <br>
oFile = 'periodicCorrelatedPatterns.txt'   #specify the output file name<br>

obj = alg.EPCPGrowth(iFile, minSup, minAllConf, maxPer, maxPerAllConf, seperator) #initialize the algorithm <br>
obj.startMine()                       #start the mining process <br>
obj.save(oFile)               #store the patterns in file <br>
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe <br>
obj.printResults()                      #Print the statistics of mining process
```

    Correlated Periodic-Frequent patterns were generated successfully using EPCPGrowth algorithm 
    Total number of Correlated Periodic-Frequent Patterns: 12
    Total Memory in USS: 115859456
    Total Memory in RSS 156979200
    Total ExecutionTime in ms: 0.0005478858947753906


The correlatedPeriodicPatterns.txt file contains the following patterns (*format:* pattern:support:lability):!cat periodicCorrelatedPatterns.txt


```python
!cat periodicCorrelatedPatterns.txt
#format- pattern:support:periodicity:allConfidence:periodicAllConfidence
```

    e:4:4:1:1 
    a:7:2:1:1 
    a	b:5:2:0.7142857142857143:1.0 
    a	d:5:3:0.625:1.5 
    a	c:6:2:0.6666666666666666:1.0 
    b:7:2:1:1 
    b	d:6:2:0.75:1.0 
    b	d	c:6:2:0.6666666666666666:1.0 
    b	c:7:2:0.7777777777777778:1.0 
    d:8:2:1:1 
    d	c:8:2:0.8888888888888888:1.0 
    c:9:2:1:1 


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
      <th>Periodicity</th>
      <th>allConf</th>
      <th>maxPerAllConf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>e</td>
      <td>4</td>
      <td>4</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>7</td>
      <td>2</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a b</td>
      <td>5</td>
      <td>2</td>
      <td>0.714286</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a d</td>
      <td>5</td>
      <td>3</td>
      <td>0.625000</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a c</td>
      <td>6</td>
      <td>2</td>
      <td>0.666667</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>b</td>
      <td>7</td>
      <td>2</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>b d</td>
      <td>6</td>
      <td>2</td>
      <td>0.750000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>b d c</td>
      <td>6</td>
      <td>2</td>
      <td>0.666667</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>b c</td>
      <td>7</td>
      <td>2</td>
      <td>0.777778</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>d</td>
      <td>8</td>
      <td>2</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>d c</td>
      <td>8</td>
      <td>2</td>
      <td>0.888889</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>c</td>
      <td>9</td>
      <td>2</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



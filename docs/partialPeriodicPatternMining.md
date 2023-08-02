# Mining Partial Periodic Patterns in Temporal Databases

## 1. What is partial periodic pattern mining?

Periodic-Frequent Pattern Mining (PFPM) is an important knowledge discovery technique in data mining with many real-world applications. It involves identifying all patterns that have exhibited perfect periodic behavior in a temporal database. A major limitation of this technique is that it fails to discover those interesting patterns that have exhibited partial periodic behavior in a temporal database. **Partial periodic pattern mining** (PPPM) has been introduced to tackle this problem.

Partial periodic pattern mining aims to discover all interesting patterns in a temporal database that satisfy the user-specified **maximum inter-arrival time (maxIAT)** and **minimum periodic support** (**minPS**) constraints. The *maxIAT* controls the maximum inter-arrival time within which a pattern must reappear in order to consider its reoccurrence periodic in a database. The *minPS* coontrols the minimum number of periodic occurrences a pattern must have in a temporal database.
 
Reference: R. Uday Kiran, Haichuan Shang, Masashi Toyoda, and Masaru Kitsuregawa. 2017. Discovering Partial Periodic Itemsets in Temporal Databases. In Proceedings of the 29th International Conference on Scientific and Statistical Database Management (SSDBM '17). Association for Computing Machinery, New York, NY, USA, Article 30, 1–6. [Link](https://doi.org/10.1145/3085504.3085535)

## 2. What is a temporal database?

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

## 4. What is the need for understanding the statistics of a database?
The performance of a pattern mining algorithm primarily depends on the satistical nature of a database. Thus, it is important to know the following details of a database:

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

## 5.  What are the input parameters to a partial periodic pattern mining algorithm?

The input parameters to a partial periodic pattern mining algorithm are: 
* __Temporal database__  <br> Acceptable formats:
> * String : E.g., 'temporalDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/temporalDatabases/temporal_T10I4D100K.csv
> * DataFrame with the header titled 'TS' and 'Transactions'

* __minPS__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __maxIAT__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __seperator__ <br> default seperator is '\t' (tab space)

## 6. How to store the output of a frequent pattern mining algorithm?
The patterns discovered by a frequent pattern mining algorithm can be saved into a file or a data frame.

## 7. How to run a partial periodic pattern mining algorithm on a terminal?

* Download the PAMI source code from [Github](https://github.com/udayRage/PAMI/archive/refs/heads/main.zip).
* Unzip the PAMI source code folder.
* Enter into periodicFrequentPattern folder.

```console
foo@bar: cd PAMI/periodicPeriodicPattern/basic
```
* Execute the python program on ther terminal using the following syntax:

```console 
foo@bar:python3 algorithmName.py inputFile outputFile minSup seperator
```

__Example:__ python3 `PPPGrowth.py` `inputFile.txt` `outputFile.txt` `4`  `3` `' '`

## 8. How to execute a partial periodic pattern mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
import PAMI.partialPeriodicPattern.basic.PPPGrowth as alg

iFile = 'sampleTemporalDatabase.txt'  #specify the input temporal database 
minPS = 5                       #specify the minPS value 
maxIAT = 2                      #specify the maxIAT value 
seperator = ' '                  #specify the seperator. Default seperator is tab space. 
oFile = 'partialPatterns.txt'   #specify the output file name

obj = alg.PPPGrowth(iFile, minPS, maxIAT, seperator) #initialize the algorithm 
obj.startMine()                       #start the mining process 
obj.save(oFile)               #store the patterns in file
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe
obj.printResults()                      #Print the statistics of mining process
```

    Partial Periodic Patterns were generated successfully using 3PGrowth algorithm 
    Total number of Partial Periodic Patterns: 9
    Total Memory in USS: 102936576
    Total Memory in RSS 142065664
    Total ExecutionTime in ms: 0.001527547836303711


The partialPatterns.txt file contains the following patterns (*format:* pattern:periodicSupport):!cat partialPatterns.txt


```terminal
!cat partialPatterns.txt
#Format is pattern:periodic-support
```

    b:6 
    b	d:5 
    b	d	c:5 
    b	c:6 
    a:6 
    a	c:5 
    d:7 
    d	c:7 
    c:8 


The dataframe containing the patterns is shown below:


```python
df  #The dataframe containing the patterns is shown below. In each pattern, items were seperated from each other with a tab space (or \t). 
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
      <td>b</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b d</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b d c</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b c</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a c</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>d</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>d c</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>c</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



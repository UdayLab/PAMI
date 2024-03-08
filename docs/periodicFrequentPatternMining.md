# Mining Periodic-Frequent Patterns in Temporal Databases

## 1. What is periodic-frequent pattern mining?

Periodic-Frequent pattern mining aims to discover all interesting patterns in a temporal database that have **support** no less than the user-specified **minimum support** (**minSup**) constraint and **periodicity** no greater than the user-specified **maximum periodicity** (**maxPer**) constraint.  The **minSup** controls the minimum number of transactions that a pattern must appear in a database and the **maxPer** controls the maximum time interval within which a pattern must reappear in the database.

Research paper: Tanbeer, Syed & Ahmed, Chowdhury & Jeong, Byeong-Soo. (2009). Discovering Periodic-Frequent Patterns in Transactional Databases. 5476. 242-253. 10.1007/978-3-642-01307-2_24 [link](https://www.researchgate.net/publication/220895259_Discovering_Periodic-Frequent_Patterns_in_Transactional_Databases/stats). 

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
import PAMI.extras.dbStats.TemporalDatabase as stats

obj = stats.TemporalDatabase('sampleTemporalDatabase.txt', ' ')
obj.run()
obj.printStats() 
```

## 5. What is the input to periodic-frequent pattern mining algorithms?

Algorithms to mine the periodic-frequent patterns requires temporal database, minSup and maxPer (specified by user).
* Temporal database is accepted following formats:
> * String : E.g., 'temporalDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/temporalDatabases/temporal_T10I4D100K.csv
> * DataFrame. Please note that dataframe must contain the header titled 'TS' and 'Transactions'
* minSup should be mentioned in 
> * __count (beween 0 to length of database)__ 
> * [0, 1]
* maxPer should be mentioned in 
> * __count (beween 0 to length of database)__ 
> * [0, 1]
* seperator  <br> default seperator is '\t' (tab space)

## 6. How to run a periodic-frequent pattern algorithm on a terminal?
* Download the PAMI source code from [Github](https://github.com/udayRage/PAMI/archive/refs/heads/main.zip).
* Unzip the PAMI source code folder.
* Enter into periodicFrequentPattern folder.

```console
foo@bar: cd PAMI/periodicFrequentPattern/basic
```
* Execute the python program on ther terminal using the following syntax:

```console 
foo@bar:python3 algorithmName.py inputFile outputFile minSup seperator
```

__Example:__ python3 `PFPGrowth.py` `inputFile.txt` `outputFile.txt` `3`  `4` `' '`

## How to execute a periodic-frequent pattern mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
import PAMI.periodicFrequentPattern.basic.PFPGrowth as alg 

iFile = 'sampleTemporalDatabase.txt'  #specify the input transactional database
minSup = 5                     #specify the minSup value
maxPer = 3                     #specify the maxPer value
seperator = ' '                #specify the seperator. Default seperator is tab space.
oFile = 'periodicFrequentPatterns.txt'   #specify the output file name

obj = alg.PFPGrowth(iFile, minSup, maxPer, seperator) #initialize the algorithm 
obj.startMine()                       #start the mining process 
obj.save(oFile)               #store the patterns in file 
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe 
obj.printResults()                      #Print the statistics of mining process
```

    Periodic Frequent patterns were generated successfully using PFPGrowth algorithm 
    Total number of Periodic Frequent Patterns: 13
    Total Memory in USS: 90112000
    Total Memory in RSS 127647744
    Total ExecutionTime in ms: 0.00040459632873535156


The periodicPatterns.txt file contains the following patterns (*format:* pattern:support:periodicity):!cat periodicPatterns.txt


```terminal
!cat periodicFrequentPatterns.txt
#Format is pattern:support:periodicity
```

    a:7:2 
    a	b:5:2 
    a	b	c:5:2 
    a	d:5:3 
    a	d	c:5:3 
    a	c:6:2 
    b:7:2 
    b	d:6:2 
    b	d	c:6:2 
    b	c:7:2 
    d:8:2 
    d	c:8:2 
    c:9:2 


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
      <th>Periodicity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a b</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a b c</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a d</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a d c</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a c</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>b</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>b d</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>b d c</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>b c</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>d</td>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>d c</td>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>c</td>
      <td>9</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



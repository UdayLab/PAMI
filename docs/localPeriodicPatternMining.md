# Mining Local Periodic Patterns in Temporal Databases

## What is local periodic pattern mining?

Local periodic pattern mining aims to discover all interesting patterns in a temporal database that have **periodicity** no greater than the user-specified **maximum periodicity** (**maxPer**) constraint, **time interval of occurence** no greater than user-specified **maximum period of spillovers** (**maxSoPer**) constraint and **minDur** is no less than **minimum duration (minDur)**.  The **minDur** controls the minimum duration that a pattern is reocurring.

Research paper: Fournier Viger, Philippe & Yang, Peng & Rage, Uday & Ventura, Sebastian & Luna, José María. (2020). Mining Local Periodic Patterns in a Discrete Sequence. Information Sciences. 544. 10.1016/j.ins.2020.09.044. 

## What is a temporal database?

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

## Acceptable format of temporal databases in PAMI

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

## Understanding the statisctics of database

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

### The sample code


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


## What is the input to local periodic pattern mining algorithms

Algorithms to mine the local periodic patterns requires temporal database, maxPer, maxSoPer and minDur (specified by user).
* Temporal database is accepted following formats:
> * String : E.g., 'temporalDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame. Please note that dataframe must contain the header titled 'TS' and 'Transactions'
* maxPer should be mentioned in 
> * __count (beween 0 to length of database)__ 
> * [0, 1]
* maxSoPer should be mentioned in 
> * __count (beween 0 to length of database)__ 
> * [0, 1]
* minDur should be mentioned in 
> * __count (beween 0 to length of database)__ 
> * [0, 1]
* seperator  <br> default seperator is '\t' (tab space)

## How to run the local periodic pattern algorithm in terminal

* Download the PAMI source code from github.
* Unzip the PAMI source code folder and enter into local periodic pattern folder.
* Enter into localPeriodicPattern folder
* Enter into a **basic** folder of your choice and execute the  following command on terminal.

__syntax:__ python3 algorithmName.py `<path to the input file>` `<path to the output file>` `<maxPer>` `<maxSoPer>` `<minDur>` `<seperator>`

## Sample command to execute the LPPGrowth algorithm in localPeriodicPattern/basic folder

python3 `LPPGrowth.py` `inputFile.txt` `outputFile.txt` 3  4  2 `' '`

## How to implement the LPPGrowth algorithm by importing PAMI package

Import the PAMI package executing:   **pip3 install PAMI**

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
import PAMI.localPeriodicPattern.basic.LPPGrowth as alg 

iFile = 'sampleTemporalDatabase.txt'  #specify the input transactional database <br>

maxPer = 3  #specify the maxPer value <br>
maxSoPer = 5  #specify the maxSoPer value <br>
minDur = 5  #specify the minDur value <br>
seperator = ' ' #specify the seperator. Default seperator is tab space. <br>
oFile = 'localPeriodicPatterns.txt'   #specify the output file name<br>

obj = alg.LPPGrowth(iFile, maxPer, maxSoPer, minDur, seperator) #initialize the algorithm <br>
obj.startMine()                       #start the mining process <br>
obj.save(oFile)               #store the patterns in file <br>
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe <br>
#obj.printStats()                      #Print the statistics of mining process
```

The localPeriodicPatterns.txt file contains the following patterns (*format:* pattern:support):!cat localPeriodicPatterns.txt


```python
!cat localPeriodicPatterns.txt
```

    f : {(4, 8)}
    ('f', 'd') : {(4, 10)}
    ('f', 'd', 'c') : {(4, 10)}
    ('f', 'c') : {(4, 10)}
    d : {(2, 10)}
    ('d', 'c') : {(2, 10)}
    ('d', 'c', 'b') : {(2, 10)}
    ('d', 'c', 'b', 'a') : {(3, 10)}
    ('d', 'c', 'a') : {(3, 10)}
    ('d', 'b') : {(2, 10)}
    ('d', 'b', 'a') : {(3, 10)}
    ('d', 'a') : {(3, 10)}
    c : {(1, 10)}
    ('c', 'b') : {(1, 10)}
    ('c', 'b', 'a') : {(1, 10)}
    ('c', 'a') : {(1, 10)}
    b : {(1, 10)}
    ('b', 'a') : {(1, 10)}
    a : {(1, 9)}


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
      <th>PTL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>f</td>
      <td>{(4, 8)}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(f, d)</td>
      <td>{(4, 10)}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(f, d, c)</td>
      <td>{(4, 10)}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(f, c)</td>
      <td>{(4, 10)}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>d</td>
      <td>{(2, 10)}</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(d, c)</td>
      <td>{(2, 10)}</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(d, c, b)</td>
      <td>{(2, 10)}</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(d, c, b, a)</td>
      <td>{(3, 10)}</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(d, c, a)</td>
      <td>{(3, 10)}</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(d, b)</td>
      <td>{(2, 10)}</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(d, b, a)</td>
      <td>{(3, 10)}</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(d, a)</td>
      <td>{(3, 10)}</td>
    </tr>
    <tr>
      <th>12</th>
      <td>c</td>
      <td>{(1, 10)}</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(c, b)</td>
      <td>{(1, 10)}</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(c, b, a)</td>
      <td>{(1, 10)}</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(c, a)</td>
      <td>{(1, 10)}</td>
    </tr>
    <tr>
      <th>16</th>
      <td>b</td>
      <td>{(1, 10)}</td>
    </tr>
    <tr>
      <th>17</th>
      <td>(b, a)</td>
      <td>{(1, 10)}</td>
    </tr>
    <tr>
      <th>18</th>
      <td>a</td>
      <td>{(1, 9)}</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

# Mining Recurring Patterns in Temporal Databases

## What is recurrent pattern mining?

Recurrent pattern mining aims to discover all interesting patterns in a temporal database that have **periodic support** no less than the user-specified **minimum periodic support** (**minPS**) constraint, **period** no greater than the user-specified **maximum Interval Time** (**maxIAT**) constraint and **Recurrence** no less than the user-specified **minimum recurrence** (**_minRec_**).  The **minSup** controls the minimum number of transactions that a pattern must appear in a database. The **maxIAT** controls the maximum interval time the pattern must reappear. The **minRec** controls the number of periodic intervals of a pattern.

Reference:  R. Uday Kiran, Haichuan Shang, Masashi Toyoda, Masaru Kitsuregawa: Discovering Recurring Patterns in Time Series. EDBT 2015: 97-108. [Link](https://openproceedings.org/2015/conf/edbt/paper-23.pdf)


## What is a temporal database?

A temporal database is an unordered collection of transactions. A temporal represents a pair constituting of temporal-timestamp and a set of items. <br> A hypothetical temporal database containing the items **_a, b, c, d, e, f, and g_**  and its timestamp is shown below

|TS| Transactions|
| --- | --- |
| 1 | a c d f g |
| 2 | a b c d |
| 3 | a c d f g | 
| 4 | a b c d e f g |
| 5 | b c e f g |
| 7 | c d e f |
| 8 | a b c d e f g | 
| 9 | c d e f g |
| 10 | a b c d e f g |
| 12 | a c d e f g |
| 13 | a c d f g |
| 14 | a c e g |
| 15 | a d f g |
| 16 | a b c |

__Note:__  Duplicate items must not exist within a transaction.

## What is the acceptable format of a temporal database in PAMI?

Each row in a temporal database must contain timestamp and items.  A sample transactional database, say sampleInputFile.txt, is provided below.


```
!cat recurringSample.txt
```

    1 a c d f g
    2 a b c d
    3 a c d f g
    4 a b c d e f g
    5 b c e f g
    7 c d e f
    8 a b c d e f g
    9 c d e f g
    10 a b c d e f g
    12 a c d e f g
    13 a c d f g
    14 a c e g
    15 a d f g
    16 a b c

## Understanding the statistics of a temporal database

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
import PAMI.extras.dbStats.TemporalDatabase as stats

obj = stats.TemporalDatabase('sampleInputFile.txt', ' ')
obj.run()
obj.printStats() 
```

## What are the input parameters?

The input parameters to a periodic frequent pattern mining algorithm are: 
* __Temporal database__  <br> Acceptable formats:
> * String : E.g., 'temporalDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'TS' and 'Transactions'

* __minSup__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __maxPer__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __minRec__  <br> specified in 
> * [0, 1]

* __seperator__ <br> default seperator is '\t' (tab space)

## How to store the output of a recurring pattern mining algorithm?
The patterns discovered by a recurring pattern mining algorithm can be saved into a file or a data frame.

## How to run the recurrent pattern mining algorithms in a terminal?

* Download the PAMI source code from github.
* Unzip the PAMI source code folder and enter into recurring pattern folder.
* Enter into recurringPattern folder
* Enter into specific folder execute the  following command on terminal.

__syntax:__ python3 algorithmName.py `<path to the input file>` `<path to the output file>` `<minSup>` `<maxPer>`  `<minRec>`  `<seperator>`

__Example:__ python3 `RPGrowth` `inputFile.txt` `outputFile.txt` `4`  `3` `2` `' '`

## How to execute a recurring pattern mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
import PAMI.recurringPattern.basic.RPGrowth as alg 

iFile = 'sampleInputFile.txt'  #specify the input temporal database <br>
minSup = 4  #specify the minSup value <br>
maxPer = 4   #specify the maxPer value <br>
minRec = 1.5    #specify the maxRec Value <br>
seperator = ' ' #specify the seperator. Default seperator is tab space. <br>
oFile = 'recurringPatterns.txt'   #specify the output file name<br>

obj = alg.RPGrowth(iFile, minSup, maxPer, minRec, seperator) #initialize the algorithm <br>
obj.startMine()                       #start the mining process <br>
obj.save(oFile)               #store the patterns in file <br>
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe <br>
obj.printResults()                      #Print the statistics of mining process
```
 


```terminal
!cat recurringPatterns.txt  # will present the patterns
```
 
```python
df #prints the data frame
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
      <th>Recurrance</th>
      <th>intervals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>11</td>
      <td>2</td>
      <td>{[1, 4] : 4}{[8, 16] : 7}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a f</td>
      <td>8</td>
      <td>2</td>
      <td>{[1, 4] : 3}{[8, 15] : 5}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a f c</td>
      <td>7</td>
      <td>2</td>
      <td>{[1, 4] : 3}{[8, 13] : 4}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a f c d</td>
      <td>7</td>
      <td>2</td>
      <td>{[1, 4] : 3}{[8, 13] : 4}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a f c d g</td>
      <td>7</td>
      <td>2</td>
      <td>{[1, 4] : 3}{[8, 13] : 4}</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a f c g</td>
      <td>7</td>
      <td>2</td>
      <td>{[1, 4] : 3}{[8, 13] : 4}</td>
    </tr>
    <tr>
      <th>6</th>
      <td>a f d</td>
      <td>8</td>
      <td>2</td>
      <td>{[1, 4] : 3}{[8, 15] : 5}</td>
    </tr>
    <tr>
      <th>7</th>
      <td>a f d g</td>
      <td>8</td>
      <td>2</td>
      <td>{[1, 4] : 3}{[8, 15] : 5}</td>
    </tr>
    <tr>
      <th>8</th>
      <td>a f g</td>
      <td>8</td>
      <td>2</td>
      <td>{[1, 4] : 3}{[8, 15] : 5}</td>
    </tr>
    <tr>
      <th>9</th>
      <td>a d</td>
      <td>9</td>
      <td>2</td>
      <td>{[1, 4] : 4}{[8, 15] : 5}</td>
    </tr>
    <tr>
      <th>10</th>
      <td>a d g</td>
      <td>8</td>
      <td>2</td>
      <td>{[1, 4] : 3}{[8, 15] : 5}</td>
    </tr>
    <tr>
      <th>11</th>
      <td>a d g c</td>
      <td>7</td>
      <td>2</td>
      <td>{[1, 4] : 3}{[8, 13] : 4}</td>
    </tr>
    <tr>
      <th>12</th>
      <td>a d c</td>
      <td>8</td>
      <td>2</td>
      <td>{[1, 4] : 4}{[8, 13] : 4}</td>
    </tr>
    <tr>
      <th>13</th>
      <td>a g</td>
      <td>9</td>
      <td>2</td>
      <td>{[1, 4] : 3}{[8, 15] : 6}</td>
    </tr>
    <tr>
      <th>14</th>
      <td>a g c</td>
      <td>8</td>
      <td>2</td>
      <td>{[1, 4] : 3}{[8, 14] : 5}</td>
    </tr>
    <tr>
      <th>15</th>
      <td>a c</td>
      <td>10</td>
      <td>2</td>
      <td>{[1, 4] : 4}{[8, 16] : 6}</td>
    </tr>
    <tr>
      <th>16</th>
      <td>d g</td>
      <td>9</td>
      <td>2</td>
      <td>{[1, 4] : 3}{[8, 15] : 6}</td>
    </tr>
    <tr>
      <th>17</th>
      <td>d g c</td>
      <td>8</td>
      <td>2</td>
      <td>{[1, 4] : 3}{[8, 13] : 5}</td>
    </tr>
    <tr>
      <th>18</th>
      <td>d g c f</td>
      <td>8</td>
      <td>2</td>
      <td>{[1, 4] : 3}{[8, 13] : 5}</td>
    </tr>
    <tr>
      <th>19</th>
      <td>d g f</td>
      <td>9</td>
      <td>2</td>
      <td>{[1, 4] : 3}{[8, 15] : 6}</td>
    </tr>
  </tbody>
</table>
</div>



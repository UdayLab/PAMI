# Mining Stable Periodic Patterns in Temporal Databases

## What is stable periodic pattern mining?

Stable periodic pattern mining aims to dicover all interesting patterns in a temporal database using three contraints **miimum support**, **maximum period** and **maximum lability**, that have **support** no less than the user-specified **minimum support** (**minSup**) constraint and **lability** no greater than **maximum lability** (**maxLa**). <br>

Reference: Fournier-Viger, P., Yang, P., Lin, J. C.-W., Kiran, U. (2019). Discovering Stable Periodic-Frequent Patterns in Transactional Data. Proc. 32nd Intern. Conf. on Industrial, Engineering and Other Applications of Applied Intelligent Systems (IEA AIE 2019), Springer LNAI, pp. 230-244

## What is a temporal database?

A temporal database is an ordered collection of transactions. A transaction represents a pair constituting of timestamp and a set of items. <br> 
A hypothetical temporal database containing the items **_a, b, c, d, e, f, and g_** is shown below

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

__Note:__  Duplicate items must not exist within a transaction.

## What is the acceptable format of a temporal database in PAMI?

Each row in a temporal database must contain timestamp and items. The stable periodic frequent pattern mining algorithms considers the timestamp to calculate the periodicity. A sample temporal database, say sampleInputFile.txt, is provided below.

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

The input parameters to a frequent pattern mining algorithm are: 
* __Temporal database__  <br> Acceptable formats:
> * String : E.g., 'transactionalDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'TS' and 'Transactions'

* __minSup__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __maxPer__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __maxLa__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __seperator__ <br> default seperator is '\t' (tab space)

## How to store the output of a stable periodic frequent pattern mining algorithm?
The patterns discovered by a stable periodic frequent pattern mining algorithm can be saved into a file or a data frame.

## How to run the stable periodic frequent pattern mining algorithms in a terminal?

* Download the PAMI source code from github.
* Unzip the PAMI source code folder and enter into stable periodic frequent pattern folder.
* Enter into stablePeriodicFrequentPattern folder
* You will find different types of folders like **basic, topk**
* Enter into **basic** folder execute the  following command on terminal.

__syntax:__ python3 algorithmName.py `<path to the input file>` `<path to the output file>` `<minSup>` `<maxPer>`  `<maxLa>`  `<seperator>`



__Example:__ python3 `SPPGrowth.py` `inputFile.txt` `outputFile.txt` `3` `4` `3` `' '`

## How to execute a stable periodic frequent pattern mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
import PAMI.stablePeriodicFrequentPattern.basic.SPPGrowth as alg 

iFile = 'sampleInputFile.txt'  #specify the input temporal database <br>
minSup = 5  #specify the minSupvalue <br>
maxPer = 3   #specify the maxPervalue <br>
maxLa = 3    #specify the minLavalue <br>
seperator = ' ' #specify the seperator. Default seperator is tab space. <br>
oFile = 'stablePatterns.txt'   #specify the output file name<br>

obj = alg.SPPGrowth(iFile, minSup, maxPer, maxLa, seperator) #initialize the algorithm <br>
obj.startMine()                       #start the mining process <br>
obj.savePatterns(oFile)               #store the patterns in file <br>
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe <br>
obj.printStats()                      #Print the stats of mining process
```

The stablePatterns.txt file contains the following patterns (*format:* pattern:support:lability):!cat stablePatterns.txt


```terminal
!cat stablePatterns.txt
```

    a :7:0 
    a b :5:0 
    a b c :5:0 
    a d :5:0 
    a d c :5:0 
    a c :6:0 
    b :7:0 
    b d :6:0 
    b d c :6:0 
    b c :7:0 
    d :8:0 
    d c :8:0 
    c :9:0 


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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a b</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a b c</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a d</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a d c</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a c</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>b</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>b d</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>b d c</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>b c</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>d</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>d c</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>c</td>
      <td>9</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



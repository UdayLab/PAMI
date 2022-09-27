# Mining Fuzzy Periodic-Frequent Patterns in Fuzzy Temporal Databases

## 1. What is fuzzy periodic-frequent pattern mining?

The fuzzy frequent pattern model aims to discover frequently occurring patterns in a fuzzy transactional database. This model implicitly assumes that the temporal occurrence information of the transactions, if any, will not affect the interestingness of a pattern to the user. However, this is seldom not the case in many real-world applications. In many applications, the user may consider a pattern occurring at periodic intervals to be more interesting than a pattern occur at aperiodic (or irregular) intervals. With this motivation, the fuzzy frequent pattern model was generalized to discover fuzzy periodic-frequent patterns in a fuzzy temporal database.

The basic model of fuzzy periodic-frequent pattern aims  to discover all patterns in a fuzzy temporal database that satisfy the user-specified   **minimum support** (**minSup**) constraint and **maximum periodicity** (**maxPer**) constraints.  The **minSup** controls the minimum number of transactions that a pattern must appear in a database. The **maxPer** controls the maximum time interval within which a pattern must reappear in the database.

Reference: R. U. Kiran et al., "Discovering Fuzzy Periodic-Frequent Patterns in Quantitative Temporal Databases," 2020 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), 2020, pp. 1-8, doi: 10.1109/FUZZ48607.2020.9177579. [Link](https://ieeexplore.ieee.org/document/9177579)

## 2. What is a fuzzy temporal database?

A fuzzy temporal database is a collection of transactions at a particular timestamp, where each transaction contains a timestamp, set of items, and its fuzzy values respectively. <br> A hypothetical fuzzy temporal database containing the items **_a, b, c, d, e, f, and g_** as shown below

| TS  | Transactions                                      |
|-----|---------------------------------------------------|
| 1   | (a.L,0.2) (b.M,0.3) (c.H,0.1) (g.M,0.1)           |
| 2   | (b.M,0.3) (c.H,0.2) (d.L,0.3) (e.H,0.2)           |
| 3   | (a.L,0.2) (b.M,0.1) (c.H,0.3) (d.L,0.4)           |
| 4   | (a.L,0.3) (c.H,0.2) (d.L,0.1) (f.M,0.2)           |
| 5   | (a.L,0.3) (b.M,0.1) (c.H,0.2) (d.L,0.1) (g.M,0.2) |
| 6   | (c.H,0.2) (d.L,0.2) (e.H,0.3) (f.M,0.1)           |
| 7   | (a.L,0.2) (b.M,0.1) (c.H,0.1) (d.L,0.2)           |
| 8   | (a.L,0.1) (e.H,0.2) (f.M,0.2)                     |
| 9   | (a.L,0.2) (b.M,0.2) (c.H,0.4) (d.L,0.2)           |
| 10  | (b.M,0.3) (c.H,0.2) (d.L,0.2) (e.H,0.2)           |

__Note:__  Duplicate items must not exist in a transaction.

## 3. What is acceptable format of a fuzzy temporal database in PAMI?

Each row in a fuzzy temporal database must contain timeStamp, list of fuzzy items, colon as a seperator, and their list of fuzzy values. <br>
- Colon ':' must be used as a seperator to distinguish fuzzy items and their fuzzy values. This seperator is fixed and cannot be over-written by the users.
- fuzzy items and fuzzy values have to be seperated from each other with a delimitar. The default delimitar is 'tab space,' however, users can over-ride this default seperator.
- There is no need for a seperator to distinct timestamp and fuzzy items. The first element in a transaction is considered as a timestamp by default. 

A sample fuzzy temporal database file, say [fuzzyTemporalDatabase.txt](fuzzyTemporalDatabase.txt), is provided below:

1 a.L b.M c.H g.M:0.2 0.3 0.1 0.1 <br>
2 b.M c.H d.L e.H:0.13 0.2 0.3 0.2 <br>
3 a.L b.M c.H d.L:0.2 0.1 0.3 0.4 <br>
4 a.L c.H d.L f.M:0.3 0.2 0.1 0.2 <br>
5 a.L b.M c.H d.L g.M:0.3 0.1 0.2 0.1 0.2 <br>
6 c.H d.L e.H f.M:0.2 0.2 0.3 0.1 <br>
7 a.L b.M c.H d.L:0.2 0.1 0.1 0.2 <br>
8 a.L e.H f.M:0.1 0.2 0.2 <br>
9 a.L b.M c.H d.H:0.2 0.2 0.4 0.2 <br>
10 b.M c.H d.L e.H:0.3 0.2 0.2 0.2 <br>


For more information on how to create a fuzzy temporal database from a quantitative (or utility) temporal database, please refer to the manual [utility2FuzzyDB.pdf](utility2FuzzyDB.pdf)

## 4. What is the need for understanding the statistics of a database?

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
import PAMI.extras.dbStats.fuzzyDatabaseStats as stats 
obj = stats.fuzzyDatabaseStats('fuzzyTemporalDatabase.txt', ' ') 
obj.run() 
obj.printStats() 
```

## 5. What are the input parameters to be specified for a fuzzy periodic-frequent pattern mining algorithm?

* __Fuzzy Temporal database__  <br> Acceptable formats:
> * String : E.g., 'fuzzyDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/fuzzyDatabases/fuzzyTemporal_T10I4D100K.csv
> * DataFrame with the header titled 'Transactions' and 'fuzzyValues'

* __minSup__  <br> specified in
> * __count (beween 0 to length of a database)__ or
> * [0, 1]
* __maxPer__  <br> specified in 
> * __count (beween 0 to length of a database)__ or
> * [0, 1]
* __seperator__ <br> default seperator is '\t' (tab space)

## 6. How to store the output of a fuzzy periodic-frequent patternn mining algorithm?
The patterns discovered by a fuzzy periodic frequent pattern mining algorithm can be saved into a file or a data frame.

## 7. How to run the fuzzy periodic-frequent pattern algorithm in a terminal?

* Download the PAMI source code from github.
* Unzip the PAMI source code folder.
* Enter into fuzzyPeriodicFrequentPattern folder.

```console
foo@bar: cd PAMI/fuzzyPeriodicFrequentPattern/basic
```

* Execute the python program on the terminal using the following syntax:

```console
foo@bar: python3 algorithmName.py inputFile outputFile minSup maxPer seperator
```

__Example:__ python3 `FPFPMiner.py` `inputFile.txt` `outputFile.txt` `3` `4` `' '`

## 8. How to execute a Fuzzy Periodic Frequent mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
import PAMI.fuzzyPeriodicFrequentPattern.basic.FPFPMiner as alg 

iFile = 'fuzzyTemporalDatabase.txt'      #specify the input temporal database 
minSup = 4                      #specify the minSup value
seperator = ' '                 #specify the seperator. Default seperator is tab space.
maxPer = 3                      #specify the maxPer value
oFile = 'FuzzyPeriodicPatterns.txt'   #specify the output file name


obj = alg.FPFPMiner(iFile, minSup, maxPer, ' ' ) 
obj.startMine() 
obj.save(oFile)           #(to store the patterns in file) 
Df = obj.getPatternsAsDataFrame() #(to store the patterns in dataframe) 
obj.printResults()                  #(to print the no of patterns, runtime and memory consumption details)
```

    Total number of Fuzzy Periodic-Frequent Patterns: 7
    Total Memory in USS: 110964736
    Total Memory in RSS 149651456
    Total ExecutionTime in seconds: 0.0006830692291259766


The FuzzyPeriodicPatterns.txt file contains the following patterns (*format:* pattern:support:periodicity):!cat FuzzyPeriodicPatterns.txt


```terminal
!cat 'FuzzyPeriodicPatterns.txt'
#format: fuzzyPeriodicPattern:support:periodicity
```

    a.L:5.4:2 
    b.L:5.6:2 
    b.L	d.L:4.199999999999999:2 
    b.L	c.L:4.6:2 
    d.L:6.199999999999999:2 
    d.L	c.L:5.4:2 
    c.L:7.0:2 


The dataframe contains the following information:


```python
Df
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
      <td>a.L</td>
      <td>5.4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b.L</td>
      <td>5.6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b.L d.L</td>
      <td>4.2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b.L c.L</td>
      <td>4.6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>d.L</td>
      <td>6.2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>d.L c.L</td>
      <td>5.4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>c.L</td>
      <td>7.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



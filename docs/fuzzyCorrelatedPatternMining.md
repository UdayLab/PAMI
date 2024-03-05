# Mining fuzzy correlated patterns in transactional database

## What is fuzzy correlated pattern mining?

Correlated pattern mining is a crucial knowledge discovery technique in big data analytics. Since the rationle of this technique is to find all interesting patterns that may exist in a **binary** transactional database, it fails to discover interesting patterns that may exist in a quantitative transactional database. To tackle this problem, *fuzzy correlated pattern mining* was introduced to discover regularities in a quantitative transactional database.

In the fuzzy correlated pattern mining, a quantiative transactional database is first transformed into a fuzzy transactional database using a set of fuzzy functions. Later, interesting patterns, called fuzzy correlated patterns, were discovered from the fuzzy transactional database using *minimum support* and *minimum all-confidence* constraints.

Reference: Lin, N.P., & Chueh, H. (2007). Fuzzy correlation rules mining. https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.416.6053&rep=rep1&type=pdf

## What is a fuzzy transactional database?

A fuzzy transactional database is a collection of transaction, where each transaction contains a set items  (or fuzzy terms) and their fuzzy values.  <br >
A hypothetical fuzzy database with items **_a, b, c, d, e, f and g_** is shown below.

| Transactions|                                     
| --- |                                              
| (a,2) (b,3) (c,1) (g,1) |                          
| (b,3) (c,2) (d,3) (e,2) |                          
| (a,2) (b,1) (c,3) (d,4) |                          
| (a,3) (c,2) (d,1) (f,2) |                          
| (a,3) (b,1) (c,2) (d,1) (g,2) |                    
| (c,2) (d,2) (e,3) (f,1) |                          
| (a,2) (b,1) (c,1) (d,2) |                          
| (a,1) (e,2) (f,2) |
| (a,2) (b,2) (c,4) (d,2) |
| (b,3) (c,2) (d,2) (e,2) |

__Note:__  Duplicate items must not exist within a transaction.

## What is the acceptable format of a fuzzy transactional database in PAMI?

Each row in a fuzzy transactional database must contain fuzzy items, a seperator, and their fuzzy values. <br>
- Colon ':' must be used as a seperator to distinguish fuzzy items and their fuzzy values. This seperator is fixed and cannot be over-written by the users.
- fuzzy items and fuzzy values have to be seperated from each other with a delimitar. The default delimitar is 'tab space,' however, users can over-ride this default seperator.

A sample fuzzy transactional database file, say fuzzyTransactionalDatabase.txt, is provided below:

a.L b.M c.H g.M:0.2 0.3 0.1 0.1 <br>
b.M c.H d.L e.H:0.13 0.2 0.3 0.2 <br>
a.L b.M c.H d.L:0.2 0.1 0.3 0.4 <br>
a.L c.H d.L f.M:0.3 0.2 0.1 0.2 <br>
a.L b.M c.H d.L g.M:0.3 0.1 0.2 0.1 0.2 <br>
c.H d.L e.H f.M:0.2 0.2 0.3 0.1 <br>
a.L b.M c.H d.L:0.2 0.1 0.1 0.2 <br>
a.L e.H f.M:0.1 0.2 0.2 <br>
a.L b.M c.H d.H:0.2 0.2 0.4 0.2 <br>
b.M c.H d.L e.H:0.3 0.2 0.2 0.2 <br>


For more information on how to create a fuzzy transactional database from a quantitative (or utility) transactional database, please refer to the manual [utility2FuzzyDB.pdf](utility2FuzzyDB.pdf)

## Understanding the statistics of a transactional database

The performance of a pattern mining algorithm primarily depends on the satistical nature of a database. Thus it is important to know the following details of a database: 
* Total number of transactions (Database size)
* Total number of unique items in database
* Minimum lenth of transaction that exists in database
* Average length of all transactions that exists in database
* Maximum length of transaction that exists in database 
* Variance in transaction length
* Sparsity of database

The below sample code prints the statistical details of a database.

```python
import PAMI.extras.dbStats.FuzzyDatabase as stats

obj = stats.FuzzyDatabase('fuzzyTransactionalDatabase.txt', ' ')
obj.run()
obj.printStats() 
```

The input parameters to a frequent pattern mining algorithm are: 
* __Fuzzy database__  <br> Acceptable formats:
> * String : E.g., 'FuzzyDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/fuzzyDatabases/fuzzy_T10I4D100K.csv
> * DataFrame with the header titled 

* __minSup__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __minAllConf__  <br> specified in 
> * [0, 1]

* __seperator__ <br> default seperator is '\t' (tab space)

## How to store the output of a fuzzy correlated pattern mining algorithm?
The patterns discovered by a fuzzy correlated pattern mining algorithm can be saved into a file or a data frame.

## How to run the fuzzy correlated pattern mining algorithms in a terminal?

* Download the PAMI source code from github.
* Unzip the PAMI source code folder and enter in to fuzzy correlated pattern
* Enter into fuzzyCorrelated pattern  folder
* Enter into the folder and execute the  following command on terminal.

__syntax:__ python3 algorithmName.py `<path to the input file>` `<path to the output file>` `<minSup>`  `<minAllConf>`  `<seperator>`

__Example:__ python3 `FCPGrowth.py` `inputFile.txt` `outputFile.txt` `4`  `0.5`  `' '`

## How to execute a fuzzy correlated pattern mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
import PAMI.fuzzyCorrelatedPattern.basic.FCPGrowth as alg 

iFile = 'sampleUtility.txt'  #specify the input temporal database <br>
minSup = 4  #specify the minSupvalue <br>    #specify the maxPerAllConfValue <br>
seperator = ' ' #specify the seperator. Default seperator is tab space. <br>
minAllConf = 0.5
oFile = 'FuzzyCorrelatedPatterns.txt'   #specify the output file name<br>

obj = alg.FCPGrowth(iFile, minSup, minAllConf, seperator) #initialize the algorithm <br>
obj.startMine()                       #start the mining process <br>
obj.save(oFile)               #store the patterns in file <br>
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe <br>
obj.printResults()                      #Print the statistics of mining process
```

The FuzzyCorrelatedPatterns.txt file contains the following patterns (*format:* pattern:support:lability):!cat FuzzyCorrelatedPatterns.txt


```terminal
!cat FuzzyCorrelatedPatterns.txt
```

    a.L : 5.3999999999999995 : 0.7714285714285714
     
    b.L : 5.599999999999999 : 0.6222222222222221
     
    b.L c.L : 4.6 : 0.5111111111111111
     
    d.L : 6.199999999999999 : 0.6888888888888888
     
    d.L c.L : 5.3999999999999995 : 0.6
     
    c.L : 6.999999999999999 : 0.7777777777777777
     


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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a.L</td>
      <td>5.3999999999999995 : 0.7714285714285714\n</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b.L</td>
      <td>5.599999999999999 : 0.6222222222222221\n</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b.L c.L</td>
      <td>4.6 : 0.5111111111111111\n</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d.L</td>
      <td>6.199999999999999 : 0.6888888888888888\n</td>
    </tr>
    <tr>
      <th>4</th>
      <td>d.L c.L</td>
      <td>5.3999999999999995 : 0.6\n</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c.L</td>
      <td>6.999999999999999 : 0.7777777777777777\n</td>
    </tr>
  </tbody>
</table>
</div>



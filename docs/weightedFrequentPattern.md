# Mining Weighted Frequent Patterns in Transactional Databases

## What is weighted frequent pattern mining?

weighted Frequent pattern mining aims to discover all interesting patterns in a transactional database that have **support** no less than the user-specified **minimum support** (**minSup**) constraint and **weight** no less than the user-specified **minimum weight** (**_minWeight_**).  The **minSup** controls the minimum number of transactions that a pattern must appear in a database. The **minWeight** controls the minimum weight of item. <br>

## What is the transactional database?

A transactional database is a collection of transactions, where each transaction contains a transaction-identifier and a set of items. <br> A hypothetical transactional database containing the items **_a, b, c, d, e, f, and g_** as shown below

|tid| Transactions|
| --- | --- |
| 1 | a c d f i m |
| 2 | a c d f m r |
| 3 | b d f m p r | 
| 4 | b c f m p |
| 5 | c d f m  r |
| 6 | d m r |

__Note:__  Duplicate items must not exist in a transaction.

## What is acceptable format of a transactional databases in PAMI

Each row in a transactional database must contain only items. The frequent pattern mining algorithms in PAMI implicitly assume the row number of a transaction as its transactional-identifier to reduce storage and processing costs. A sample transactional database, say [WFIMSample.txt](WFIMSample.txt), is provided below.

a c d f i m <br>
a c d f m r <br>
b d f m p r <br>
b c f m p  <br>
c d f m  r  <br>
d m r  <br>

## What is the Weighted database?

A weight database is a collection of items with their weights. <br> 
A hypothetical weight database, say [WFIMWeightSample.txt](WFIMWeightSample.txt), containing the items **_a, b, c, d, e, f, and g_** as shown below:

a 1.3  <br>
b 1.1  <br>
c 1.4  <br>
d 1.2  <br>
f 1.5  <br>
i 1.1  <br>
m 1.3  <br>
p 1.0  <br>
r 1.5  <br>

## Understanding the statisctics of database

To understand about the database. The below code will give the detail about the transactional database.
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
import PAMI.extras.dbStats.transactionalDatabaseStats as stats 
obj = stats.transactionalDatabaseStats('WFIMSample.txt', ' ') 
obj.run() 
obj.printStats() 
```

The input parameters to a frequent pattern mining algorithm are: 
* __Transactional database__  <br> Acceptable formats:
> * String : E.g., 'transactionalDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'Transactions'

* __minSup__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __minWeight__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __seperator__ <br> default seperator is '\t' (tab space)

## How to store the output of a weighted frequent pattern mining algorithm?
The patterns discovered by a correlated pattern mining algorithm can be saved into a file or a data frame.

## How to run the weighted frequent pattern mining algorithms in a terminal?

* Download the PAMI source code from github.
* Unzip the PAMI source code folder and enter into weighted frequent pattern folder.
* Enter into weightedFrequentPattern folder
* Enter into a specific folder and execute the  following command on terminal.

__syntax:__ python3 algorithmName.py `<path to the input file>` `<path to the output file>` `<path to the weight file>` `<minSup>` `<minWeight>` `<seperator>`

## Sample command to execute the WFIM code in weightedFrequentPattern folder

__Example:__ python3 `WFIM.py` `inputFile.txt` `outputFile.txt` `weightSample.txt` `3` `2` `' '`

## How to execute a weighted frequent pattern mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
import PAMI.weightedFrequentPattern.WFIM as alg 

iFile = 'WFIMSample.txt'  #specify the input transactional database <br>
wFile = 'WFIMWeightSample.txt'  #specify the input transactional database <br>
minSup = 3  #specify the minSupvalue <br>
minWeight = 1.2    #specify the minWeight value <br>
seperator = ' ' #specify the seperator. Default seperator is tab space. <br>
oFile = 'weightedPatterns.txt'   #specify the output file name<br>

obj = alg.WFIM(iFile, wFile, minSup, minWeight, seperator) #initialize the algorithm <br>
obj.startMine()                       #start the mining process <br>
obj.savePatterns(oFile)               #store the patterns in file <br>
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe <br>
obj.printStats()                      #Print the statistics of mining process
```

The weightedPatterns.txt file contains the following patterns (*format:* pattern:support): !cat weightedPatterns.txt


```python
!cat weightedPatterns.txt
```

    r :4 
    r d :4 
    r d m :4 
    r m :4 
    c :4 
    c f :4 
    c f m :4 
    c m :4 
    f :5 
    f d :4 
    f d m :4 
    f m :5 
    d :5 
    d m :5 
    m :6 


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
      <td>r</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>r d</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>r d m</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>r m</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c f</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>c f m</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>c m</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>f</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>f d</td>
      <td>4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>f d m</td>
      <td>4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>f m</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>d</td>
      <td>5</td>
    </tr>
    <tr>
      <th>13</th>
      <td>d m</td>
      <td>5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>m</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



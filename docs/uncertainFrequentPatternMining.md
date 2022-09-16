# Mining Frequent Patterns in Uncertain Transactional Databases

## What is frequent pattern mining?

Frequent pattern mining aims to discover all interesting patterns in a transactional database that have **support** no less than the user-specified **minimum support** (**minSup**) constraint.  The **minSup** controls the minimum number of transactions that a pattern must appear in a database.

## What is a uncertain transactional database?

A transactional database is a collection of transactions, where each transaction contains a transaction-identifier and a set of items with their respective uncertain value. <br> A hypothetical transactional database containing the items **_a, b, c, d, e, f, and g_** as shown below.

|tid| Transactions|
| --- | --- |
| 1 | a(0.4) b(0.5) c(0.2) g(0.1)  |
| 2 | b(0.2) c(0.3) d(0.4) e(0.2) |
| 3 | a(0.3) b(0.1) c(0.3) d(0.4) | 
| 4 | a(0.2) c(0.6) d(0.2) f(0.1) |
| 5 | a(0.3) b(0.2) c(0.4) d(0.5) g(0.3) |
| 6 | c(0.2) d(0.7) e(0.34) f(0.2) |
| 7 | a(0.6) b(0.4) c(0.3) d(0.2) |
| 8 | a(0.2) e(0.2) f(0.2)  | 
| 9 | a(0.1) b(0.3) c(0.2) d(0.4) |
| 10 | b(0.3) c(0.2) d(0.1) e(0.6) |

__Note:__  Duplicate items must not exist in a transaction.

## What is the acceptable format of a uncertain transactional databases in PAMI?

Each row in a transactional database must contain only items with their respective uncertain values.  A sample transactional database, say sampleInputFile.txt, is provided below.

a(0.4) b(0.5) c(0.2) g(0.1)   <br> 
b(0.2) c(0.3) d(0.4) e(0.2)   <br>
a(0.3) b(0.1) c(0.3) d(0.4)   <br>
a(0.2) c(0.6) d(0.2) f(0.1)   <br>
a(0.3) b(0.2) c(0.4) d(0.5) g(0.3)  <br> 
c(0.2) d(0.7) e(0.34) f(0.2)  <br>
a(0.6) b(0.4) c(0.3) d(0.2)   <br>
a(0.2) e(0.2) f(0.2)    <br>
a(0.1) b(0.3) c(0.2) d(0.4)   <br>
b(0.3) c(0.2) d(0.1) e(0.6)   <br>

## What are the input parameters?

The input parameters to a frequent pattern mining algorithm are: 
* __Transactional database__  <br> Acceptable formats:
> * String : E.g., 'uncertainTransactionalDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'Transactions'

* __minSup__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]
* __seperator__ <br> default seperator is '\t' (tab space)

## How to store the output of a frequent pattern mining algorithm?
The patterns discovered by a frequent pattern mining algorithm can be saved into a file or a data frame.

## How to run the frequent pattern mining algorithms in a terminal?


* Download the PAMI source code from github.
* Unzip the PAMI source code folder and enter into uncertain frequent pattern folder.
* Enter into uncertainFrequentPattern folder
* You will find folder like **basic**
* Enter into a specific folder and execute the  following command on terminal.

__syntax:__ python3 algorithmName.py `<path to the input file>` `<path to the output file>` `<minSup>` `<seperator>`

__Example:__ python3 `PUFGrowth.py` `inputFile.txt` `outputFile.txt` 0.05 `' '`

## How to execute a frequent pattern mining algorithm in a Jupyter Notebook?

Import the PAMI package executing:   **pip3 install PAMI**

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
import PAMI.uncertainFrequentPattern.basic.PUFGrowth as alg 

iFile = 'sampleInputFile.txt'  #specify the input transactional database <br>
minSup = 0.5                  #specify the minSup value <br>
seperator = ' '                #specify the seperator. Default seperator is tab space. <br>
oFile = 'frequentPatterns.txt'   #specify the output file name<br>

obj = alg.PUFGrowth(iFile, minSup, seperator) #initialize the algorithm <br>
obj.startMine()                       #start the mining process <br>
obj.savePatterns(oFile)               #store the patterns in file <br>
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe <br>
obj.printStats()                      #Print the statistics of mining process
```

The frequentPatterns.txt file contains the following patterns (*format:* pattern:support):!cat frequentPatterns.txt


```python
!cat frequentPatterns.txt
```

f  0.5 <br>
e  1.3399999999999999  <br>
b  2.0  <br>
b a  0.56  <br>
b c  0.51   <br>
a  2.099999999999996  <br>
a c  0.6100000000000001  <br>
c  2.7   <br>
d  2.9000000000000004 <br>
c d  0.8600000000000001  <br>

The dataframe containing the patterns is shown below:


```python
df
```

|  | Patterns | Support |
| --- | --- | --- |
| 0 | f | 0.50 |
| 1 | e | 1.34 |
| 2 | b | 2.00 | 
| 3 | b a | 0.56 |
| 4 | b c | 0.51 |
| 5 | a | 2.09 |
| 6 | a c | 0.61 |
| 7 | c | 2.70 | 
| 8 | d | 2.90 |
| 9 | c d | 0.86 |

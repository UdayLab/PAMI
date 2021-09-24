**[CLICK HERE](index.html)** to access the PAMI manual.


# Theoretical representation of a transactional database

A transactional database is a collection of transactions.  Every transaction constitutes of a transaction-identifier (TID)
and a set of items. A sample transactional database generated from the set of items, I={a,b,c,d,e,f}, is shown in below table:

  TID |  Transactions 
     --- | -----
     1   | a, b, c
     2   | d, e
     3   | a, e, f
   
### Rules to create a transactional database
1. Since TID of a transaction directly represents its row number in a database, we can ignore this information 
to save storage space and processing time. 
1. All items in every transaction must be with a separator.   

### Transactional database format
The format of a transactional database is as follows:

      item1<sep>item2<sep>...<sep>itemN

      An example:
        a   b   c
        d   e
        a   e   f

**Note:**
1. The default separator used in PAMI is tab space (or \t). However, the users can user any separator of their choice, such as space and comma. Since spatial objects, such as Point, Line, and Polygon, are represented using space 
   and comma, usage of tab space facilitates us to effectively distinguish the spatial objects.
1. In a transactional database, items can be represented in integers or strings.

### Example: finding frequent patterns in a transactional database using FP-growth
1. Execute the following command if PAMI was not installed in your machine.
   
         pip install pami
   
1. [Click here](https://www.u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv) to download the synthetic T10I4D100K transactional database.
1. Move the downloaded 'transactional_T10I4D100K.csv' file  into a directory, say /home/userName.
1. Change your present working directory to /home/userName
1. Copy and paste the below code in a python file, say testPAMI.py
   
   ```Python
   from PAMI.frequentPattern.basic import fpGrowth as alg
  
   inputFile = '/home/userName/transactional_T10I4D100K.csv' 
   outputFile = '/home/userName/patterns.txt'
   minSup = 10.0   # 10 percentage
   
   obj = alg.fpGrowth(inputFile, minSup) 
   #use obj = alg.fpGrowth(inputFile, minSup,sep=',')  to override the default tab space separator with comma
   obj.startMine()  #start the mining process
   obj.savePatterns(outputFile)      #store the generated patterns in a file
      

   ```
1. Execute the testPAMI.py file by typing the following command

       python3 testPAMI.py
       
1. After the successful execution, users will find the generated patterns in patterns.txt file

 

# Theoretical representation of a transactional database

A transactional database generally records binary data. Every transaction in this database 
   contains a transaction-identifier (tid) and a set of items. The format of a transaction is as follows:
   
         tid : items
   
  A sample transactional database generated from the set of items, I={a,b,c,d,e,f}, is shown in below table:

  TID |  Transactions 
     --- | -----
     1   | a,b,c
     2   | d,e
     3   | a,e,f

   **NOTE:** Since TID of a transaction directly represents its row number in a database, we can ignore this information 
   to save storage space and processing time.
   
# Practical representation of a transactional database in PAMI
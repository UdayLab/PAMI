# Theoretical representation

 A temporal database generally records binary data. Every transaction in this database
   contains a timestamp (ts) and a set of items. The format of a transaction in this database is as follows:
         timestamp : Items
   
   A sample temporal database generated from the set of items, I={a,b,c,d,e,f}, is shown in below table:
   
   Timestamp |  Transactions 
     --- | -----
     1   | a,b,c
     2   | d,e
     5   | a,e,f 

# Practical representation of a temporal database in PAMI
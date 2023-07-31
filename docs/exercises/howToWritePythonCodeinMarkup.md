
   ```Python
   from PAMI.frequentPattern.basic import FPGrowth as alg
  
   inputFile = '/home/userName/transactional_T10I4D100K.csv' 
   outputFile = '/home/userName/patterns.txt'
   minSup = 10.0   # 10 percentage
   
   obj = alg.fpGrowth(inputFile, minSup) 
# use obj = alg.fpGrowth(inputFile, minSup,sep=',')  to override the default tab space separator with comma
obj.startMine()  # start the mining process
obj.save(outputFile)      # store the generated patterns in a file


   ```



### Example: finding periodic-frequent patterns in a transactional database using PFP-growth++
1. Execute the following command if PAMI was not installed in your machine.

         pip install pami

1. [Click here](https://www.u-aizu.ac.jp/~udayrage/datasets/temporalDatabases/temporal_T10I4D100K.csv) to download the synthetic T10I4D100K temporal database.
1. Move the downloaded 'temporal_T10I4D100K.csv' file  into a directory, say /home/userName.
1. Change your present working directory to /home/userName
1. Copy and paste the below code in a python file, say testPAMI.py

   ```Python
   from PAMI.periodicFrequentPattern.basic import PFPGrowthPlus as alg
  
   inputFile = '/home/userName/temporal_T10I4D100K.csv' 
   outputFile = '/home/userName/temporalPatterns.txt'
   minSup = 10.0   # 10 percentage of the transactions
   maxPer = 1.0    # 1 percentage of the transactions
   
   obj = alg.PFPGrowthPlus(inputFile, minSup, maxPer) 
   
   #use obj = alg.fpGrowth(inputFile, minSup,sep=',')  to override the default tab space separator with comma
   obj.startMine()  #start the mining process
   obj.save(outputFile)      #store the generated patterns in a file
      

   ```
1. Execute the testPAMI.py file by typing the following command
   python3 testPAMI.py
1. After the successful execution, users will find the generated patterns in temporalPatterns.txt file



## Example: finding high utility patterns in a utility transactional database using EFIM
1. Execute the following command if PAMI was not installed in your machine.

         pip install pami

1. [Click here](https://www.u-aizu.ac.jp/~udayrage/datasets/temporalDatabases/utility_T10I4D100K.csv) to download the synthetic T10I4D100K temporal database.
1. Move the downloaded 'utility_T10I4D100K.csv' file  into a directory, say /home/userName.
1. Change your present working directory to /home/userName
1. Copy and paste the below code in a python file, say testPAMI.py

   ```Python
   from PAMI.highUtilityPattern.basic import EFIM as alg
  
   inputFile = '/home/userName/utility_T10I4D100K.csv' 
   outputFile = '/home/userName/utilityPatterns.txt'
   minUtil = 10000
   
   obj = alg.EFIM(inputFile,    minUtil) 
   
   obj.startMine()  #start the mining process
   obj.save(outputFile)      #store the generated patterns in a file
      

   ```
1. Execute the testPAMI.py file by typing the following command
   python3 testPAMI.py
1. After the successful execution, users will find the generated patterns in utilityPatterns.txt file

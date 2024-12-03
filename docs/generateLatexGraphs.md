# **[Home](index.html) | [Exercises](exercises.html) | [Real-world Examples](examples.html)**  


[Exceuting Algorithms in PAMI](utilization.html)    
   1. [Importing PAMI algorithms into your program](useAlgo.html)
   
          from PAMI.frequentPattern.basic import FPGrowth  as alg
          
          obj = alg.FPGrowth(inputFile,minSup,sep)
          obj.mine()
          obj.save('patterns.txt')
          df = obj.getPatternsAsDataFrame()
          print('Runtime: ' + str(obj.getRuntime()))
          print('Memory: ' + str(obj.getMemoryRSS()))

   2. [Evaluation multiple algorithms with multiple minSup values and storing it in DataFrame](useAlgo.md)
         
            import pandas as pd
            from PAMI.extras import generateLatexGraphFile as ab
            result = pd.DataFrame(columns=['algorithm', 'minSup', 'patterns', 'runtime', 'memory'])

            from PAMI.frequentPattern.basic import FPGrowth as alg
            dataset = 'https://www.u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv'
            minSupList = [3000,2000,1500,1000]
            algorithm = 'FPGrowth'
            for minSup in minSupList:
               obj = alg.FPGrowth(dataset, minSup=minSup)
               obj.mine()
               df = pd.DataFrame([algorithm, minSup, len(obj.getPatterns()), obj.getRuntime(), obj.getMemoryRSS()], index=result.columns).T
               result = result.append(df, ignore_index=True)
    
            from PAMI.frequentPattern.basic import ECLAT as alg
            dataset = 'https://www.u-aizu.ac.jp/~udayrage/datasets/temporalDatabases/temporal_T10I4D100K.csv'
            minSupList = [3000,2000,1500,1000]
            algorithm = 'ECLAT'
            for minSup in minSupList:
               obj = alg.ECLAT(dataset, minSup=minSup)
               obj.mine()
               df = pd.DataFrame([algorithm, minSup, len(obj.getPatterns()), obj.getRuntime(), obj.getMemoryRSS()], index=result.columns).T
               result = result.append(df, ignore_index=True)
    
            print(result)
            ab.generateLatexCode(result)
         
   3. [The sample result of evaluation of above code](useAlgo.md)

               algorithm minSup patterns     runtime      memory
            0  FPGrowth   3000       60   22.811239   718159872
            1  FPGrowth   2000      155  358.451235  1000931328
            2  FPGrowth   1500      237   37.498651  1307623424
            3  FPGrowth   1000      385   29.919302  1588002816
            4     ECLAT   3000       60    18.06122  1796489216
            5     ECLAT   2000      155   24.058386  2028683264
            6     ECLAT   1500      237   19.436389  2020114432
            7     ECLAT   1000      386   25.425784  2029191168


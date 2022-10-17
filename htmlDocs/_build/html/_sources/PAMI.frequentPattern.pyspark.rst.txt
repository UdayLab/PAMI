PAMI.frequentPattern.pyspark package
====================================

Submodules
----------

PAMI.frequentPattern.pyspark.parallelApriori module
---------------------------------------------------

.. automodule:: PAMI.frequentPattern.pyspark.parallelApriori
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 parallelApriori.py <inputFile> <outputFile> <minSup> <numWorkers>

        Example:
                  >>>  python3 parallelApriori.py sampleDB.txt patterns.txt 10.0 3

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

            import PAMI.frequentPattern.pyspark.parallelApriori as alg

            obj = alg.parallelApriori(iFile, minSup, numWorkers)

            obj.startMine()

            frequentPatterns = obj.getPatterns()

            print("Total number of Frequent Patterns:", len(frequentPatterns))

            obj.savePatterns(oFile)

            Df = obj.getPatternInDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)


**Credits:**

         The complete program was written by Yudai Masu  under the supervision of Professor Rage Uday Kiran.


PAMI.frequentPattern.pyspark.parallelECLAT module
-------------------------------------------------

.. automodule:: PAMI.frequentPattern.pyspark.parallelECLAT
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>> python3 parallelECLAT.py <inputFile> <outputFile> <minSup> <numWorkers>

        Example:
                  >>> python3 parallelECLAT.py sampleDB.txt patterns.txt 10.0 3

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

            import PAMI.frequentPattern.pyspark.parallelECLAT as alg

            obj = alg.parallelECLAT(iFile, minSup, numWorkers)

            obj.startMine()

            frequentPatterns = obj.getPatterns()

            print("Total number of Frequent Patterns:", len(frequentPatterns))

            obj.savePatterns(oFile)

            Df = obj.getPatternInDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)


**Credits:**

         The complete program was written by Yudai Masu under the supervision of Professor Rage Uday Kiran.


PAMI.frequentPattern.pyspark.parallelFPGrowth module
----------------------------------------------------

.. automodule:: PAMI.frequentPattern.pyspark.parallelFPGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>> python3 parallelFPGrowth.py <inputFile> <outputFile> <minSup> <numWorkers>

        Example:
                  >>>  python3 parallelFPGrowth.py sampleDB.txt patterns.txt 10.0 3

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

                import PAMI.frequentPattern.pyspark.parallelFPGrowth as alg

                obj = alg.parallelFPGrowth(iFile, minSup, numWorkers)

                obj.startMine()

                frequentPatterns = obj.getPatterns()

                print("Total number of Frequent Patterns:", len(frequentPatterns))

                obj.savePatterns(oFile)

                Df = obj.getPatternInDataFrame()

                memUSS = obj.getMemoryUSS()

                print("Total Memory in USS:", memUSS)

                memRSS = obj.getMemoryRSS()

                print("Total Memory in RSS", memRSS)

                run = obj.getRuntime()

                print("Total ExecutionTime in seconds:", run)


**Credits:**

         The complete program was written by Yudai Masu  under the supervision of Professor Rage Uday Kiran.


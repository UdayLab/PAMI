PAMI.correlatedPattern.basic package
====================================

Submodules
----------

PAMI.correlatedPattern.basic.CPGrowth module
--------------------------------------------

.. automodule:: PAMI.correlatedPattern.basic.CPGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 CPGrowth.py <inputFile> <outputFile> <minSup> <minAllConf> <sep>

        Example:
                  >>>  python3 CPGrowth.py inp.txt output.txt 4.0 0.3

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.correlatedPattern.basic import CPGrowth as alg

        obj = alg.CPGrowth(iFile, minSup,minAllConf)

        obj.startMine()

        correlatedPatterns = obj.getPatterns()

        print("Total number of correlated frequent Patterns:", len(correlatedPatterns))

        obj.savePatterns(oFile)

        Df = obj.getPatternInDf()

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  Sai Chitra.B  under the supervision of Professor Rage Uday Kiran.


PAMI.correlatedPattern.basic.CPGrowthPlus module
------------------------------------------------

.. automodule:: PAMI.correlatedPattern.basic.CPGrowthPlus
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 CPGrowthPlus.py <inputFile> <outputFile> <minSup> <minAllConf> <sep>

        Example:
                  >>>  python3 CPGrowthPlus.py sampleDB.txt patterns.txt 0.23 0.2

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.correlatedPattern.basic import CPGrowthPlus as alg

        obj = alg.CPGrowthPlus(iFile, minSup,minAllConf)

        obj.startMine()

        correlatedPatterns = obj.getPatterns()

        print("Total number of correlated frequent Patterns:", len(correlatedPatterns))

        obj.savePatterns(oFile)

        Df = obj.getPatternInDf()

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)


**Credits:**

         The complete program was written by  Sai Chitra.B  under the supervision of Professor Rage Uday Kiran.



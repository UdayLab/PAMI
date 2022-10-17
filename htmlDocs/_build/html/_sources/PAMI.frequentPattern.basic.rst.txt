PAMI.frequentPattern.basic package
==================================

Submodules
----------

PAMI.frequentPattern.basic.Apriori module
-----------------------------------------

.. automodule:: PAMI.frequentPattern.basic.Apriori
   :members:
   :undoc-members:
   :show-inheritance:





**Methods to execute code on terminal**

        Format:
                  >>> python3 Apriori.py <inputFile> <outputFile> <minSup>

        Example:
                  >>>  python3 Apriori.py sampleDB.txt patterns.txt 10.0

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

         import PAMI.frequentPattern.basic.Apriori as alg

         obj = alg.Apriori(iFile, minSup)

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

         The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PAMI.frequentPattern.basic.ECLAT module
---------------------------------------

.. automodule:: PAMI.frequentPattern.basic.ECLAT
   :members:
   :undoc-members:
   :show-inheritance:

**Methods to execute code on terminal**

        Format:
                  >>> python3 ECLAT.py <inputFile> <outputFile> <minSup>

        Example:
                  >>>  python3 ECLAT.py sampleDB.txt patterns.txt 10.0

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        import PAMI.frequentPattern.basic.ECLAT as alg

        obj = alg.ECLAT(iFile, minSup)

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

         The complete program was written by Kundai  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PAMI.frequentPattern.basic.ECLATDiffset module
----------------------------------------------

.. automodule:: PAMI.frequentPattern.basic.ECLATDiffset
   :members:
   :undoc-members:
   :show-inheritance:

**Methods to execute code on terminal**

        Format:
                  >>> python3 ECLATDiffset.py <inputFile> <outputFile> <minSup>

        Example:
                  >>> python3 ECLATDiffset.py sampleDB.txt patterns.txt 10.0

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

            import PAMI.frequentPattern.basic.ECLATDiffset as alg

            obj = alg.ECLATDiffset(iFile, minSup)

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

           The complete program was written by Kundai under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PAMI.frequentPattern.basic.ECLATbitset module
---------------------------------------------

.. automodule:: PAMI.frequentPattern.basic.ECLATbitset
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>> python3 ECLATbitset.py <inputFile> <outputFile> <minSup>

        Example:
                  >>> python3 ECLATbitset.py sampleDB.txt patterns.txt 10.0

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

            import PAMI.frequentPattern.basic.ECLATbitset as alg

            obj = alg.ECLATbitset(iFile, minSup)

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

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PAMI.frequentPattern.basic.FPGrowth module
------------------------------------------

.. automodule:: PAMI.frequentPattern.basic.FPGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>> python3 FPGrowth.py <inputFile> <outputFile> <minSup>

        Example:
                  >>> python3 FPGrowth.py sampleDB.txt patterns.txt 10.0

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

            from PAMI.frequentPattern.basic import FPGrowth as alg

            obj = alg.FPGrowth(iFile, minSup)

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

           The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.


+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


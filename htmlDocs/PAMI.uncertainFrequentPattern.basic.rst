PAMI.uncertainFrequentPattern.basic package
===========================================

Submodules
----------

PAMI.uncertainFrequentPattern.basic.CUFPTree module
---------------------------------------------------

.. automodule:: PAMI.uncertainFrequentPattern.basic.CUFPTree
   :members:
   :undoc-members:
   :show-inheritance:

**Methods to execute code on terminal**

        Format:
                  >>>  python3 CUFPTree.py <inputFile> <outputFile> <minSup>
        Example:
                  >>>  python3 CUFPTree.py sampleTDB.txt patterns.txt 3

         .. note:: minSup  will be considered in support count or frequency


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.uncertainFrequentPattern.basic import CUFPTree as alg

        obj = alg.CUFPTree(iFile, minSup)v

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.savePatterns(oFile)

        Df = obj.getPatternsAsDataFrame()

        memUSS = obj.getmemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)
        
**Credits:**

         The complete program was written by  P.Likhitha  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


PAMI.uncertainFrequentPattern.basic.PUFGrowth module
----------------------------------------------------

.. automodule:: PAMI.uncertainFrequentPattern.basic.PUFGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>> python3 PUFGrowth.py <inputFile> <outputFile> <minSup>
        Example:
                  >>>  python3 PUFGrowth.py sampleTDB.txt patterns.txt 3

        .. note:: minSup  will be considered in support count or frequency


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.uncertainFrequentPattern.basic import puf as alg

        obj = alg.PUFGrowth(iFile, minSup)

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.savePatterns(oFile)

        Df = obj.getPatternsAsDataFrame()

        memUSS = obj.getmemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)
        
**Credits:**

         The complete program was written by  P.Likhitha  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


PAMI.uncertainFrequentPattern.basic.TUFP module
-----------------------------------------------

.. automodule:: PAMI.uncertainFrequentPattern.basic.TUFP
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>> python3 TUFP.py <inputFile> <outputFile> <minSup>
        Example:
                  >>>  python3 TUFP.py sampleTDB.txt patterns.txt 0.6

        .. note:: minSup  will be considered in support count or frequency

**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.uncertainFrequentPattern.basic import TUFP as alg

        obj = alg.TUFP(iFile, minSup)

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.savePatterns(oFile)

        Df = obj.getPatternsAsDataFrame()

        memUSS = obj.getmemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)
        
**Credits:**

         The complete program was written by   P.Likhitha   under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


PAMI.uncertainFrequentPattern.basic.TubeP module
------------------------------------------------

.. automodule:: PAMI.uncertainFrequentPattern.basic.TubeP
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>> python3 TubeP.py <inputFile> <outputFile> <minSup>
        Example:
                  >>>  python3 TubeP.py sampleTDB.txt patterns.txt 3

        .. note:: minSup  will be considered in support count or frequency

**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.uncertainFrequentPattern.basic import TubeP as alg

        obj = alg.TubeP(iFile, minSup)

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.savePatterns(oFile)

        Df = obj.getPatternsAsDataFrame()

        memUSS = obj.getmemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)
        
**Credits:**

         The complete program was written by  P.Likhitha  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


PAMI.uncertainFrequentPattern.basic.TubeS module
------------------------------------------------

.. automodule:: PAMI.uncertainFrequentPattern.basic.TubeS
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>> python3 TubeS.py <inputFile> <outputFile> <minSup>
        Example:
                  >>>  python3 TubeS.py sampleTDB.txt patterns.txt 3

        .. note:: minSup  will be considered in support count or frequency

**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.uncertainFrequentPattern.basic import TubeS as alg

        obj = alg.TubeS(iFile, minSup)

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.savePatterns(oFile)

        Df = obj.getPatternsAsDataFrame()

        memUSS = obj.getmemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)
        
**Credits:**

         The complete program was written by  P.Likhitha   under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


PAMI.uncertainFrequentPattern.basic.UFGrowth module
---------------------------------------------------

.. automodule:: PAMI.uncertainFrequentPattern.basic.UFGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 PUFGrowth.py <inputFile> <outputFile> <minSup>
        Example:
                  >>>  python3 PUFGrowth.py sampleTDB.txt patterns.txt 3

        .. note:: minSup  will be considered in support count or frequency

**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.uncertainFrequentPattern.basic import UFGrowth as alg

        obj = alg.UFGrowth(iFile, minSup)

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.savePatterns(oFile)

        Df = obj.getPatternsAsDataFrame()

        memUSS = obj.getmemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)
        
**Credits:**

         The complete program was written by P.Likhitha under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


PAMI.uncertainFrequentPattern.basic.UVECLAT module
--------------------------------------------------

.. automodule:: PAMI.uncertainFrequentPattern.basic.UVECLAT
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>> python3 uveclat.py <inputFile> <outputFile> <minSup>
        Example:
                  >>>  python3 uveclat.py sampleTDB.txt patterns.txt 3

        .. note:: minSup  will be considered in support count or frequency

**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.uncertainFrequentPattern.basic import UVECLAT as alg

        obj = alg.UVEclat(iFile, minSup)


        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.savePatterns(oFile)

        Df = obj.getPatternsAsDataFrame()

        memUSS = obj.getmemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)
        
**Credits:**

         The complete program was written by   P.Likhitha    under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



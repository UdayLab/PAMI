PAMI.highUtilityPatterns.basic package
======================================

Submodules
----------

PAMI.highUtilityPatterns.basic.EFIM module
------------------------------------------

.. automodule:: PAMI.highUtilityPatterns.basic.EFIM
   :members:
   :undoc-members:
   :show-inheritance:



**Methods to execute code on terminal**

        Format:
                  >>>  python3 EFIM.py <inputFile> <outputFile> <minUtil> <sep>
        Example:
                  >>>  python3 EFIM sampleTDB.txt output.txt 35

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.highUtilityPatterns.basic import EFIM as alg

        obj=alg.EFIM("input.txt",35)

        obj.startMine()

        Patterns = obj.getPatterns()

        print("Total number of high utility Patterns:", len(Patterns))

        obj.savePatterns("output")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by   pradeep pallikila  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



PAMI.highUtilityPatterns.basic.HMiner module
--------------------------------------------

.. automodule:: PAMI.highUtilityPatterns.basic.HMiner
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 HMiner.py <inputFile> <outputFile> <minUtil> <separator>
        Example:
                  >>>  python3 HMiner.py sampleTDB.txt output.txt 35

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.highUtilityPatterns.basic import HMiner as alg

        obj = alg.HMiner("input.txt",35)

        obj.startMine()

        Patterns = obj.getPatterns()

        print("Total number of high utility Patterns:", len(Patterns))

        obj.savePatterns("output")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  B.Sai Chitraa  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


PAMI.highUtilityPatterns.basic.UPGrowth module
----------------------------------------------

.. automodule:: PAMI.highUtilityPatterns.basic.UPGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 UPGrowth <inputFile> <outputFile> <Neighbours> <minUtil> <sep>
        Example:
                  >>>  python3 UPGrowth sampleTDB.txt output.txt sampleN.txt 35

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.highUtilityPatterns.basic import UPGrowth as alg

        obj=alg.UPGrowth("input.txt",35)

        obj.startMine()

        highUtilityPatterns = obj.getPatterns()

        print("Total number of Spatial Frequent Patterns:", len(highUtilityPatterns))

        obj.savePatterns("output")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by pradeep pallikila  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


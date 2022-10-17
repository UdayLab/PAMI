PAMI.highUtilityFrequentPatterns.basic package
==============================================

Submodules
----------

PAMI.highUtilityFrequentPatterns.basic.HUFIM module
---------------------------------------------------

.. automodule:: PAMI.highUtilityFrequentPatterns.basic.HUFIM
   :members:
   :undoc-members:
   :show-inheritance:

**Methods to execute code on terminal**

        Format:
                  >>>  python3 HUFIM.py <inputFile> <outputFile> <minUtil> <sep>
        Example:
                  >>>  python3 HUFIM.py sampleTDB.txt output.txt 35 20

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.highUtilityFrequentPatterns.basic import HUFIM as alg

        obj=alg.HUFIM("input.txt", 35, 20)

        obj.startMine()

        Patterns = obj.getPatterns()

        print("Total number of high utility frequent Patterns:", len(Patterns))

        obj.savePatterns("output")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  Pradeep Pallakila  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


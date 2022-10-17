PAMI.highUtilityFrequentSpatialPattern.basic package
====================================================

Submodules
----------

PAMI.highUtilityFrequentSpatialPattern.basic.SHUFIM module
----------------------------------------------------------

.. automodule:: PAMI.highUtilityFrequentSpatialPattern.basic.SHUFIM
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>> python3 SHUFIM.py <inputFile> <outputFile> <Neighbours> <minUtil> <minSup> <sep>
        Example:
                  >>> python3 SHUFIM.py sampleTDB.txt output.txt sampleN.txt 35 20

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.highUtilityFrequentSpatialPattern.basic import SHUFIM as alg

        obj=alg.SHUFIM("input.txt","Neighbours.txt",35,20)

        obj.startMine()

        patterns = obj.getPatterns()

        print("Total number of Spatial high utility frequent Patterns:", len(patterns))

        obj.savePatterns("output")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  Pradeep pallaikila  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


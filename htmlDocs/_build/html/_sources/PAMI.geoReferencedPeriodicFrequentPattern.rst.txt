PAMI.geoReferencedPeriodicFrequentPattern package
=================================================

Submodules
----------

PAMI.geoReferencedPeriodicFrequentPattern.GPFPMiner module
----------------------------------------------------------

.. automodule:: PAMI.geoReferencedPeriodicFrequentPattern.GPFPMiner
   :members:
   :undoc-members:
   :show-inheritance:

**Methods to execute code on terminal**

        Format:
                  >>>  python3 GPFPMiner.py <inputFile> <outputFile> <neighbourFile> <minSup> <maxPer>
        Example:
                  >>>  python3 GPFPMiner.py sampleTDB.txt output.txt sampleN.txt 0.5 0.3

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        import PAMI.geoReferencedPeridicFrequentPattern.GPFPMiner as alg

        obj = alg.GPFPMiner("sampleTDB.txt", "sampleN.txt", 5, 3)

        obj.startMine()

        Patterns = obj.getPatterns()

        print("Total number of Geo Referenced Periodic-Frequent Patterns:", len(Patterns))

        obj.savePatterns("outFile")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  P. Likhitha  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



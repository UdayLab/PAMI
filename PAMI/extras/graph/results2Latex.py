# results2Latex is used to convert the given results into LatexFile.
#
# **Importing this algorithm into a python program**
#
#             from PAMI.extras.graph import results2Latex
#
#             obj = results2Latex()
#
#             obj.print(resultsDF, xaxis='minSup', yaxis='patterns', label='algorithm')
#
#             obj.print(resultsDF, xaxis='minSup', yaxis='runtime', label='algorithm')
#
#             obj.print(resultsDF, xaxis='minSup', yaxis='memoryRSS', label='algorithm')
#
#             obj.print(resultsDF, xaxis='minSup', yaxis='memoryUSS', label='algorithm')
#
#             obj.save(resultsDF, xaxis='minSup', yaxis='patterns', label='algorithm', oFile='patterns.tex')
#
#             obj.save(resultsDF, xaxis='minSup', yaxis='runtime', label='algorithm', oFile='runtime.tex')
#
#             obj.save(resultsDF, xaxis='minSup', yaxis='memoryRSS', label='algorithm', oFile='memoryRSS.tex')
#
#             obj.save(resultsDF, xaxis='minSup', yaxis='memoryUSS', label='algorithm', oFile='memoryUSS.tex')
#

__copyright__ = """
Copyright (C)  2021 Rage Uday Kiran

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""



import pandas as pd

class results2Latex:
    """
    **About this algorithm**

    :**Description**:  generateLatexCode is used to convert the given DataFrame into a LaTeX file.

    :**Reference**:

    **Attributes:** - **latexCode** (*str*) -- Stores the generated LaTeX code.

    **Methods:**    - **generateLatexCode(result: pd.DataFrame, xColumn, yColumn, algorithmColumn=None)** -- *Generates LaTeX code based on DataFrame columns.*

                    - **print(result: pd.DataFrame, xaxis, yaxis, label=None)** -- *Prints the LaTeX code.*

                    - **save(result: pd.DataFrame, xaxis, yaxis, label=None, oFile="output.tex")** -- *Saves the LaTeX code to a .tex file.*

    **Importing this algorithm into a Python program**

    .. code-block:: python

            from PAMI.extras.graph import results2Latex

            resultsDF = Dataframe

            obj = results2Latex()

             obj.print(resultsDF, xaxis='minSup', yaxis='patterns', label='algorithm')

             obj.print(resultsDF, xaxis='minSup', yaxis='runtime', label='algorithm')

             obj.print(resultsDF, xaxis='minSup', yaxis='memoryRSS', label='algorithm')

             obj.print(resultsDF, xaxis='minSup', yaxis='memoryUSS', label='algorithm')

             obj.save(resultsDF, xaxis='minSup', yaxis='patterns', label='algorithm', oFile='patterns.tex')

             obj.save(resultsDF, xaxis='minSup', yaxis='runtime', label='algorithm', oFile='runtime.tex')

             obj.save(resultsDF, xaxis='minSup', yaxis='memoryRSS', label='algorithm', oFile='memoryRSS.tex')

             obj.save(resultsDF, xaxis='minSup', yaxis='memoryUSS', label='algorithm', oFile='memoryUSS.tex')

"""

    def __init__(self):
        self.latexCode = ""

    def generateLatexCode(self, result: pd.DataFrame, xColumn, yColumn, algorithmColumn=None) -> None:
        if result.empty:
            raise ValueError("The input DataFrame is empty. Please provide a DataFrame with data.")

        requiredColumns = [xColumn, yColumn]
        if algorithmColumn:
            requiredColumns.append(algorithmColumn)
        for col in requiredColumns:
            if col not in result.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

        titles = [xColumn, yColumn]
        legendary = pd.unique(result[algorithmColumn] if algorithmColumn else result.iloc[:, 0].values.ravel())
        xaxisValues = result[xColumn].values
        yaxisValues = result[yColumn].values
        algo = result[algorithmColumn].values if algorithmColumn else result.iloc[:, 0].values
        xLabel = titles[0]
        colors = ['red', 'blue', 'green', 'black', 'yellow']
        latexCode = ""
        latexCode += "\\begin{axis}[\n\txlabel={\\Huge{" + xLabel + "}},"
        latexCode += "\n\tylabel={\\Huge{" + titles[1] + "}},"
        latexCode += "\n\txmin=" + str(min(xaxisValues)) + ", xmax=" + str(max(xaxisValues)) + ",]\n"

        for num, legend in enumerate(legendary):
            color = colors[num % len(colors)]
            latexCode += "\\addplot+  [" + color + "]\n\tcoordinates {\n"
            for x, y, a in zip(xaxisValues, yaxisValues, algo):
                if legend == a:
                    latexCode += f"({x},{y})\n"
            latexCode += "\t};   \\addlegendentry{" + str(legend) + "}\n"
        latexCode += "\\end{axis}"
        self.latexCode = latexCode

    def print(self, result: pd.DataFrame, xaxis, yaxis, label=None):
        """
        Print the LaTeX code with user-specified parameters.
        """
        self.generateLatexCode(result, xaxis, yaxis, label)
        print(self.latexCode)

    def save(self, result: pd.DataFrame, xaxis, yaxis, label=None, oFile="output.tex"):
        """
        Save the generated LaTeX code to a file with user-specified parameters.
        """
        self.generateLatexCode(result, xaxis, yaxis, label)
        with open(oFile, "w") as latexWriter:
            latexWriter.write(self.latexCode)
        print(f"LaTeX file saved as {oFile}")



"""data = {
    'algorithm': ['Alg1', 'Alg2', 'Alg3', 'Alg4'],
    'minSup': [200, 400, 600, 800],
    'patterns': [800, 600, 400, 200],
    'runtime': [107.7, 98.6, 86.8, 12.5],
    'memoryRSS': [16, 15, 14, 13],
    'memoryUSS': [26, 25, 24, 23]
}
resultsDF = pd.DataFrame(data)
obj = results2Latex()

obj.print(resultsDF, xaxis='minSup', yaxis='patterns', label='algorithm')
obj.print(resultsDF, xaxis='minSup', yaxis='runtime', label='algorithm')
obj.print(resultsDF, xaxis='minSup', yaxis='memoryRSS', label='algorithm')
obj.print(resultsDF, xaxis='minSup', yaxis='memoryUSS', label='algorithm')

obj.save(resultsDF, xaxis='minSup', yaxis='patterns', label='algorithm', oFile='patterns.tex')
obj.save(resultsDF, xaxis='minSup', yaxis='runtime', label='algorithm', oFile='runtime.tex')
obj.save(resultsDF, xaxis='minSup', yaxis='memoryRSS', label='algorithm', oFile='memoryRSS.tex')
obj.save(resultsDF, xaxis='minSup', yaxis='memoryUSS', label='algorithm', oFile='memoryUSS.tex')"""




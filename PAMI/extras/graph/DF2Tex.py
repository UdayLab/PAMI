# DF2Tex is used to convert the given dataframe into LatexFile.
#
# **Importing this algorithm into a python program**
#
#             from PAMI.extras.graph import DF2Tex
#
#             obj = DF2Tex()
#
#             obj.generateLatexCode(result, "minSup", "runtime", "algorithmColumn")
#
#             obj.printLatex()
#
#             obj.save("outputFile.tex")
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

class DF2Tex:
    """
    **About this algorithm**

    :**Description**:  generateLatexFileFromDataFrame is used to convert the given DataFrame into a LaTeX file.

    :**Reference**:

    **Attributes:** - **latexCode** (*str*) -- Stores the generated LaTeX code.

    **Methods:**    - **generateLatexCode(result: pd.DataFrame, xColumn, yColumn, algorithmColumn=None)** -- *Generates LaTeX code based on DataFrame columns.*

                    - **print_latex()** -- *Prints the LaTeX code.*

                    - **save(outputFileName: str)** -- *Saves the LaTeX code to a .tex file.*

    **Importing this algorithm into a Python program**

    .. code-block:: python

            from PAMI.extras.graph import DF2Tex

            result = Dataframe

            obj = DF2Tex()

            obj.generateLatexCode(result, "minSup", "runtime", "algorithmColumn")

            obj.printLatex()

            obj.save("outputFile.tex")
    """

    def __init__(self):
        self.latexCode = ""

    def generateLatexCode(self, result: pd.DataFrame, xColumn, yColumn, algorithmColumn=None) -> None:
        """
        Generate LaTeX code from a given DataFrame.
        """
        # Check if DataFrame is empty
        if result.empty:
            raise ValueError("The input DataFrame is empty. Please provide a DataFrame with data.")

        # Check if required columns exist
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
            color = colors[num % len(colors)]  # Ensure color index is within range
            latexCode += "\\addplot+  [" + color + "]\n\tcoordinates {\n"
            for x, y, a in zip(xaxisValues, yaxisValues, algo):
                if legend == a:
                    latexCode += f"({x},{y})\n"
            latexCode += "\t};   \\addlegendentry{" + str(legend) + "}\n"
        latexCode += "\\end{axis}"
        self.latexCode = latexCode

    def printLatex(self):
        """
        Print the generated LaTeX code.
        """
        print(self.latexCode)

    def save(self, outputFileName):
        """
        Save the generated LaTeX code to a file.

        :param outputFileName: The name of the output .tex file.
        :raises ValueError: If LaTeX code has not been generated.

        """
        if not self.latexCode:
            raise ValueError("LaTeX code is empty. Please generate it before saving.")
        with open(outputFileName, "w") as latexwriter:
            latexwriter.write(self.latexCode)
        print(f"LaTeX file saved as {outputFileName}")

# Example usage
# result = pd.DataFrame()
# obj = DF2Tex()
# generateLatexCode function as four parameters dataFrame, xColumn-name, yColumn-name,
# algorithmColumn-name is optional
# obj.generateLatexCode(result, "minSup", "runtime", "algorithmColumn")
# printLatexCode function prints the output of latex file
# obj.printLatex()
# save function gives the outputFile
# obj.save("outputFile.tex")

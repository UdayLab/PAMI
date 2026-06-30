# plotAssociationRuleScatter visualizes association rules as a support vs. measure scatter plot.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.extras.graph import plotAssociationRuleScatter as ars
#
#     obj = ars.plotAssociationRuleScatter(df)
#
#     obj.plot()
#
#     obj.save(oFile='ruleScatter.png')
#


__copyright__ = """
Copyright (C)  2026 Rage Uday Kiran

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

import warnings
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


class plotAssociationRuleScatter:
    """
    Scatter plot of association rules: one point per rule, positioned by support
    and an interestingness measure, optionally coloured by a third measure.

    The input is a DataFrame from a PAMI AssociationRules miner. Measure columns
    are auto-detected when not specified explicitly.

    :Attributes:

        dataFrame : pandas.DataFrame
            association rules with a support column and at least one measure column
        xColumn : str
            column for the x-axis (default Support)
        yColumn : str
            column for the y-axis (auto-detected when omitted)
        colorColumn : str
            column mapped to point colour (auto-detected when omitted; uniform if none)

    :Methods:

        plot()
            display the scatter plot interactively
        save(oFile)
            save the scatter plot to a file
        getStatistics()
            print rule count and measure ranges

    **Importing this algorithm into a python program**
    --------------------------------------------------------
    .. code-block:: python

            from PAMI.extras.graph import plotAssociationRuleScatter as ars

            obj = ars.plotAssociationRuleScatter(df)

            obj.plot()

            obj.save(oFile='ruleScatter.png')
    """

    _MEASURES = ['Confidence', 'Lift', 'Leverage']

    def __init__(self, dataFrame: pd.DataFrame, xColumn: str = 'Support',
                 yColumn: Optional[str] = None, colorColumn: Optional[str] = None) -> None:
        """
        :param dataFrame: rules from a PAMI AssociationRules miner
        :type dataFrame: pandas.DataFrame
        :param xColumn: column name for the x-axis
        :type xColumn: str
        :param yColumn: column name for the y-axis (auto-detected when None)
        :type yColumn: str or None
        :param colorColumn: column name for point colour (auto-detected when None)
        :type colorColumn: str or None
        """
        self.dataFrame = dataFrame
        self.xColumn = xColumn
        self._userYColumn = yColumn
        self._userColorColumn = colorColumn
        self.yColumn = yColumn
        self.colorColumn = colorColumn

    def __repr__(self) -> str:
        nRules = len(self.dataFrame) if self.dataFrame is not None else 0
        return (f"plotAssociationRuleScatter(rules={nRules}, "
                f"x={self.xColumn!r}, y={self.yColumn!r}, "
                f"color={self.colorColumn!r})")

    def _resolveColumns(self) -> None:
        """
        Decide which columns drive the y-axis and point colour. Re-resolves
        from the live DataFrame each time so that post-construction mutations
        are handled correctly.
        """
        if self.dataFrame is None or self.dataFrame.empty:
            return

        present = [m for m in self._MEASURES if m in self.dataFrame.columns]

        yColumn = self._userYColumn
        colorColumn = self._userColorColumn

        if yColumn is None:
            yColumn = present[0] if present else None
        if colorColumn is None:
            remaining = [m for m in present if m != yColumn]
            colorColumn = remaining[0] if remaining else None

        self.yColumn = yColumn
        self.colorColumn = colorColumn

    def _renderFigure(self) -> bool:
        """
        Create a matplotlib figure for the scatter plot.
        Returns False when there is nothing to draw.
        """
        df = self.dataFrame
        if df is None or df.empty:
            warnings.warn("No association rules to plot.")
            return False

        self._resolveColumns()

        if self.xColumn not in df.columns:
            warnings.warn(f"x-axis column '{self.xColumn}' not found in DataFrame "
                          f"(available: {list(df.columns)})")
            return False
        if self.yColumn is None:
            warnings.warn(f"No measure column found (expected one of: {self._MEASURES})")
            return False
        if self.yColumn not in df.columns:
            warnings.warn(f"y-axis column '{self.yColumn}' not found in DataFrame "
                          f"(available: {list(df.columns)})")
            return False

        plt.figure()
        if self.colorColumn is not None and self.colorColumn in df.columns:
            sc = plt.scatter(df[self.xColumn], df[self.yColumn],
                             c=df[self.colorColumn], cmap='viridis', alpha=0.7)
            plt.colorbar(sc, label=self.colorColumn)
        else:
            if self.colorColumn is not None:
                warnings.warn(f"Colour column '{self.colorColumn}' not found; "
                              f"using uniform colour.")
            plt.scatter(df[self.xColumn], df[self.yColumn], c='tab:blue', alpha=0.7)
        plt.xlabel(self.xColumn)
        plt.ylabel(self.yColumn)
        plt.title(f"{self.yColumn} vs {self.xColumn}")
        plt.tight_layout()
        return True

    def plot(self) -> None:
        """Display the scatter plot interactively."""
        if self._renderFigure():
            plt.show()

    def save(self, oFile: str = 'ruleScatter.png') -> None:
        """
        Save the scatter plot to a file.

        :param oFile: output file path
        :type oFile: str
        """
        if self._renderFigure():
            plt.savefig(oFile)
            plt.close()
            print(f"Association-rule scatter saved as {oFile}!")

    def getStatistics(self) -> None:
        """Print rule count and the range of each available measure."""
        df = self.dataFrame
        if df is None or df.empty:
            print("No association rules.")
            return
        print("Statistics:")
        print(f"  Number of rules: {len(df)}")
        for col in [self.xColumn] + self._MEASURES:
            if col in df.columns:
                print(f"  {col}: min={df[col].min():.4f}  max={df[col].max():.4f}")


if __name__ == "__main__":
    sample = pd.DataFrame([
        {"Antecedent": "milk",  "Consequent": "bread",  "Support": 0.40, "Confidence": 0.80, "Lift": 1.6},
        {"Antecedent": "bread", "Consequent": "butter", "Support": 0.30, "Confidence": 0.60, "Lift": 1.2},
        {"Antecedent": "milk",  "Consequent": "butter", "Support": 0.25, "Confidence": 0.50, "Lift": 0.9},
        {"Antecedent": "jam",   "Consequent": "bread",  "Support": 0.20, "Confidence": 0.90, "Lift": 2.1},
    ])
    obj = plotAssociationRuleScatter(sample)
    obj.save('sampleRuleScatter.png')
    obj.getStatistics()
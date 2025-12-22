# sortedPatternFrequencyGraph
#
# Generates a Sorted Pattern Frequency graph
# showing how often each mined frequent pattern occurs in the dataset.
#
# X-axis: Patterns sorted by frequency (most frequent → least frequent)
# Y-axis: Support count of each pattern
#
# The graph visualizes the "long tail" distribution among discovered patterns.
# It also supports plotting up to five minimum support thresholds for comparison.
#
# Usage Example:
# from PAMI.extras.graph import sortedPatternFrequencyGraph as spfg
# spfg.generateSortedPatternFrequencyGraph("Transactional_T10I4D100K.csv", "\t", "Apriori", [100, 200, 300])
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

import importlib
import plotly.graph_objects as go

def generateSortedPatternFrequencyGraph(inputFile, sep="\t", algorithm="Apriori", minSupList=None):
    print(f"Generating Graph using {algorithm}...")

    if minSupList is None:
        minSupList = [100]

    if len(minSupList) > 5:
        print("Warning: Only up to 5 minimum supports are supported; truncating list.")
        minSupList = minSupList[:5]

    #Import chosen algorithm dynamically
    try:
        module = importlib.import_module(f"PAMI.frequentPattern.basic.{algorithm}")
        AlgoClass = getattr(module, algorithm)
    except Exception as e:
        print("Error importing algorithm:", e)
        return

    #Prepare Plotly figure
    fig = go.Figure()

    #Use different colors for multiple minsups
    colors = ["blue", "red", "green", "orange", "yellow"]

    for idx, minSup in enumerate(minSupList):
        try:
            print(f"Mining frequent patterns with minSup = {minSup}...")
            obj = AlgoClass(inputFile, minSup, sep)
            obj.mine()
            patterns = obj.getPatterns()
        except Exception as e:
            print(f"Error running algorithm for minSup={minSup}:", e)
            continue

        if not patterns:
            print(f"No frequent patterns found for minSup={minSup}.")
            continue

        #Sort patterns by support count (descending order)
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        pattern_labels = [
            "{" + ", ".join(p) + "}" if isinstance(p, (list, tuple, set)) else str(p)
            for p, _ in sorted_patterns
        ]
        supports = [v for _, v in sorted_patterns]

        fig.add_trace(go.Scatter(
            x=list(range(1, len(patterns) + 1)),
            y=supports,
            mode="lines+markers",
            line=dict(color=colors[idx % len(colors)], width=2),
            marker=dict(size=5, color=colors[idx % len(colors)]),
            name=f"minSup = {minSup}",
            hovertemplate=(
                "<b>Rank:</b> %{x}<br>"
                f"<b>MinSup:</b> {minSup}<br>"
                "<b>Pattern:</b> %{text}<br>"
                "<b>Support Count:</b> %{y}<extra></extra>"
            ),
            text=pattern_labels
        ))

    fig.update_layout(
        title=f"Sorted Pattern Frequency Graph ({algorithm})",
        xaxis=dict(
            title="Pattern Rank (sorted by support count)",
            showspikes=True,
            showline=True,
            mirror=True,
            linewidth=2,
            linecolor="black",
            ticks="outside",
            tickwidth=1.5,
            tickcolor="black",
            tickfont=dict(size=12, color="black")
        ),
        yaxis=dict(
            title="Support Count",
            showspikes=True,
            showline=True,
            mirror=True,
            linewidth=2,
            linecolor="black",
            ticks="outside",
            tickwidth=1.5,
            tickcolor="black",
            tickfont=dict(size=12, color="black")
        ),
        hovermode="closest",
        hoverlabel=dict(bgcolor="white", font=dict(size=12, color="black")),
        legend=dict(
            x=1.05, y=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            title="Minimum Supports"
        ),
        margin=dict(r=160, t=60, l=70, b=60),
        template="plotly_white",
        font=dict(size=13)
    )

    fig.show()

if __name__ == "__main__":
    inputFile = "Transactional_T10I4D100K.csv"
    generateSortedPatternFrequencyGraph(inputFile, "\t", "Apriori", [1000, 2000, 3000, 4000])
# Network Analysis of the Imperial Japanese Army Medical Corps, 1931–1945

Replication package for the article *"Three Reflexive Diagnostics for Historical Network
Analysis: A Methodological Essay on the Imperial Japanese Army Medical Corps,
1931–1945"* (author / venue / year — to be completed).

## What this is

Code, data and an interactive viewer for an **affiliation (prosopographical) network**
of the Imperial Japanese Army medical corps, together with three reflexive diagnostics:

1. **Edge-definition sensitivity** — Louvain community detection plus an edge-type
   ablation (removing shared-specialty edges) to test whether the observed clusters
   are structure in the sources or an artifact of how the network was built.
2. **Archival silence** — whether figures known from outside the network to be central
   (Unit 731 leadership, for instance) appear peripheral or absent, and what that
   cannot by itself establish.
3. **Sample-boundary audit** — restricting the population to actors demonstrably
   active in 1931–1945 and recording the inclusion and exclusion rules.

It also runs a **school-homophily permutation test** (5,000 permutations, seed 42, both
unrestricted and conditional on primary specialty) and writes an interactive HTML viewer
of the geo-topology layout.

## Contents

| Path | Description |
|---|---|
| `japan_medical_network_army.py` | Full pipeline: ingestion, normalisation, network construction, statistics, viewer |
| `verify_paper_numbers.py` | Reproduces every figure in the article from the released CSVs and reports PASS / FAIL |
| `japan_medical_network.html` | Interactive viewer; also embeds the node metadata in a `var meta = {...}` block |
| `nodes.csv` | 444 nodes with attributes, degree, betweenness rank, community, 731 marker |
| `edges.csv` | 8,024 undirected edges with weight and channel type |
| `school_alias_map.csv` | Orthographic normalisation of school names |
| `major_alias_map.csv` | Normalisation of specialty labels |
| `README_data.md` | Column definitions and the headline figures to reproduce |

## Data

Compiled by the author from **published sources**: some two hundred institutional
histories (大学史誌 / 部局史誌) and military medical serials (陸軍軍医団雑誌,
陸軍軍医学校防疫研究報告), cross-validated against the biographical dictionary
日本近現代医学人名事典. After orthographic normalisation (旧字体 → 新字体), homonym
disambiguation, specialty-label normalisation and temporal and branch filtering, the
verified Army-side population is **N = 483**, of whom **444 share at least one specialty
or unit affiliation and form the analytic network (8,024 undirected edges)**; the
remaining 39 are reported separately as isolates.

All individuals are deceased historical figures documented in published works. The data
contain no living-subject or otherwise sensitive information, and the node table, edge
list and both alias maps are released in full.

## Requirements

Running the pipeline:

    pip install pandas numpy networkx scipy python-louvain pyvis openpyxl

Python 3.9 or later. The pipeline reads the source spreadsheets, which are not
redistributed here, and calls a translation endpoint for Japanese labels.

Running the verification script requires **nothing beyond the standard library** — it
works from the released CSVs alone and contains its own Louvain implementation:

    python verify_paper_numbers.py

## Reproducing the article

    python verify_paper_numbers.py     # checks all reported figures against the CSVs

This is the recommended entry point for a reader or reviewer. It rebuilds the network
statistics from `nodes.csv` and `edges.csv` and prints a PASS / FAIL line for
each claim, including the permutation tests, assortativity coefficients, community
structure, the edge-ablation diagnostic and Table 2.

To rebuild the network from the original rosters instead:

    python japan_medical_network_army.py

## A note on language

Comments and console messages are in English throughout. Some Korean string literals
remain in `japan_medical_network_army.py`: the source spreadsheets use a mixture of
Korean and Japanese column headers, so those literals are data keys and cannot be
translated without breaking ingestion. They are flagged in the file.

## Citation

(author). (year). Three Reflexive Diagnostics for Historical Network Analysis:
A Methodological Essay on the Imperial Japanese Army Medical Corps, 1931–1945.
(journal). DOI: (to be added)

## License

Code: MIT. Data: CC BY 4.0.

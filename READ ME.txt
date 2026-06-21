# Network Analysis of the Imperial Japanese Army Medical Corps, 1931–1945

Replication code for the article *"Three Reflexive Diagnostics for Historical Network
Analysis: A Methodological Essay on the Imperial Japanese Army Medical Corps,
1931–1945"* (author / venue / year — to be completed).

## What this is

Analysis code that builds an **affiliation (prosopographical) network** of the
Imperial Japanese Army medical corps and applies three reflexive diagnostics:

1. **Structural robustness** — Louvain community detection + an edge-type ablation
   (removing shared-specialty edges) to test whether clusters are real structure or
   an artifact of how the network was built.
2. **Data robustness ("archival silence")** — whether figures known to be central
   from outside the network (e.g. Unit 731 leadership) are peripheral/absent.
3. **Temporal robustness** — controls the sample to actors active in 1931–1945.

It also runs a **homophily permutation test** (school ties, n = 500) and exports an
interactive HTML viewer of the geo-topology layout.

## Data

Compiled by the author from **published sources**: ~200 institutional histories
(大学史誌 / 部局史誌), military medical serials (陸軍軍医団雑誌, 陸軍軍医学校防疫研究報告),
cross-validated against the biographical dictionary 日本近現代医学人名事典. After
orthographic normalisation (旧字体→新字体), homonym disambiguation, and temporal/branch
filtering, the **verified Army medical population is N = 438 (4,295 weighted edges)**.

The dataset derives entirely from published works and contains no living-subject or
sensitive data. **It is available from the author on reasonable request** (or place
`nodes.csv` / `edges.csv` here if you choose to release it openly).

## Requirements

Python 3.9+ with: pandas, numpy, networkx, scipy, python-louvain (community), pyvis, openpyxl

    pip install pandas numpy networkx scipy python-louvain pyvis openpyxl

## How to run

    python japan_medical_network_army.py

Reads the source data from the configured paths, builds the Army-only network, prints
the diagnostic figures used in the article, and writes an interactive HTML viewer.

## Citation

(author). (year). Three Reflexive Diagnostics for Historical Network Analysis:
A Methodological Essay on the Imperial Japanese Army Medical Corps, 1931–1945.
(journal). DOI: (to be added)

## License

Code: MIT (suggested). Data (if released): CC BY 4.0 (suggested).

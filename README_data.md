# Replication data — Imperial Japanese Army medical network (1931–1945)

This folder contains the node and edge data underlying the article's network analysis,
so that the homophily test, community detection, and centrality measures can be
independently reproduced. All persons are deceased historical figures compiled from
published sources; the data contain no living-subject or private information.

## Files

### `nodes.csv` (N = 444)
The verified Army medical population that enters the network (isolates removed).
Columns:
- `node_id` — internal identifier (matches `edges.csv`).
- `name` — name (kanji).
- `school` — alma mater (node attribute; **not** used to generate edges).
- `major` — medical specialty (edge criterion + attribute).
- `military_unit` — assigned unit / epidemic-prevention body (edge criterion + attribute).
- `degree` — number of career-overlap ties (degree centrality).
- `betweenness_rank` — rank by betweenness centrality (1 = highest).
- `community` — Louvain community id (resolution = 1.0).
- `unit731` — 1 if the record carries a 731 / epidemic-prevention-unit marker.

### `edges.csv` (E = 8,024, undirected)
One row per undirected tie. Columns:
- `source`, `target` — node ids.
- `weight` — 1 (single shared channel) or 2 (specialty **and** unit shared).
- `type` — `MAJ` (shared specialty only, 7,236), `UNIT` (shared unit only, 710),
  `BOTH` (both, 78).

### `school_alias_map.csv`
Orthographic normalization table used to merge spelling variants (old/new kanji,
abbreviations, promotions) before edge construction. Reported homophily depends on this
normalization (see the article's Methods), so the table is released for audit.

## Reproducing the headline numbers
With `nodes.csv` + `edges.csv`:
- School homophily (unconditional label permutation, 5,000×): observed 13.6 %, null 11.1 %, z = 2.83, p = 0.0064.
- Conditional permutation (school labels shuffled within specialty blocks): z = 2.33, p = 0.019.
- School assortativity (Newman categorical): r = 0.002; specialty assortativity r = 0.82.
- Synthetic check (specialty–school correlation injected, no true school homophily): unconditional permutation z ≈ 139, conditional (within-specialty) permutation z ≈ 0.
- Tokyo Imperial University share: 122 / 444 (27.5 %). Kyoto 56, Keio 37, Kyushu 31, Hokkaido 21.
- Density 0.082; modularity 0.63 → 0.55 after removing shared-specialty edges.
- Louvain (seed 42, resolution 1.0): **19 communities**, matching the `community` column of
  `nodes.csv`. At resolution 0.5 and 2.0 the count becomes 19 and 31 respectively.
- Within-community specialty concentration (sum of each community's modal primary specialty
  divided by total members): **71.7 %** on the full projection, **30.0 %** on the unit-only
  network. Removing the 7,314 specialty-related edges leaves 788 unit-related edges among
  103 persons, partitioned into 12 communities.

## Specialty-label normalization (please read before re-running)

Specialty strings are normalized before edges are built, mirroring the school alias map.
The rules are released as `major_alias_map.csv` and implemented in `major_list_en()` in
`japan_medical_network_army.py`:

1. **Case unification.** Machine translation returned the same specialty with different
   capitalization (`Internal medicine` / `internal medicine`, `surgery` / `Surgery`, …);
   eighteen such pairs occur.
2. **Parenthetical repair.** `Internal medicine (cardiology, hepatology)` was previously
   split on the comma into two fragments. Parenthetical detail is now stripped before
   splitting, and the record folds into its parent specialty.
3. **Separator fix.** `/` and `;` now split multiple specialties (`Surgery/Anesthesiology`).
4. **Synonym merging.** `orthopedics` → `orthopedic surgery`, `public health science` →
   `public health`, `allergy` → `allergology`, `clinical laboratory (medicine)` →
   `laboratory medicine`, `medical school education` → `medical education`.
5. **Non-specialty exclusion.** Attributes that are not medical specialties (poet, painter,
   politician, businessman, activist, …) and bare occupational labels (`doctor`, `army doctor`,
   `nurse`) are retained as node attributes but do **not** generate specialty edges.

This reduces 107 raw labels to 60. Three persons in the verified set carry no specialty label
after normalization. The analytic network is 444 nodes (39 isolates reported separately),
against 452 / 31 under the earlier unnormalized convention.

## Interactive viewer

`japan_medical_network.html` is the self-contained interactive viewer (geo-topology layout,
community colouring, per-person relation badges). It embeds the same node metadata used above
in a `var meta = {...}` block, so the network can also be reconstructed directly from it.
Visualization-only controls (hiding or filtering edges on screen) do not affect any computed
statistic.

A verification script (`verify_paper_numbers.py`) and the full pipeline
(`japan_medical_network_army.py`) are provided alongside.

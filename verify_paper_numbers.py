#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_paper_numbers.py
=======================
Reproduces every quantitative claim in the article from the released replication
data and reports PASS / FAIL against the values printed in the paper.

    Three Reflexive Diagnostics for Historical Network Analysis:
    The Imperial Japanese Army Medical Corps, 1931-1945

WHAT IT READS
    nodes.csv   444 nodes  (id, name, school, major, military_unit, degree,
                            betweenness_rank, community, unit731)
    edges.csv   8,024 undirected edges (source, target, weight, type)

    Either beside this script or in a data/ subdirectory; both layouts work.

    Nothing else. No source spreadsheets, no translation API, no internet.

HOW TO RUN
    python verify_paper_numbers.py

    Run it from the directory that holds the CSV files.
    Standard library only - no pandas, networkx, or python-louvain required.
    A multilevel Louvain implementation is included below so that community
    detection is reproducible without external packages.

WHAT IT DOES NOT DO
    It does not rebuild the network from the original rosters. Edge construction,
    orthographic normalisation of school names (`data/school_alias_map.csv`) and
    of specialty labels (`data/major_alias_map.csv`) happen in
    `japan_medical_network_army.py`. This script takes the released edge list as
    given and checks that every statistic in the article follows from it.

    Community detection depends on the implementation and on the random seed.
    The `community` column of nodes.csv is the partition reported in the article;
    the Louvain run below is expected to match it in cluster count and modularity,
    but a different library may return a slightly different partition.
"""

import csv
import itertools
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

SEED = 42
N_PERM = 5000
HERE = os.path.dirname(os.path.abspath(__file__))


def _find(name):
    """Locate a data file whether it sits in ./data/ or beside this script."""
    for cand in (os.path.join(HERE, "data", name), os.path.join(HERE, name)):
        if os.path.exists(cand):
            return cand
    return os.path.join(HERE, "data", name)


# ----------------------------------------------------------------------------
# Specialty-label normalisation - identical to major_list_en() in the pipeline.
# Released as major_alias_map.csv.
# ----------------------------------------------------------------------------
MAJOR_NON_SPECIALTY = {
    "poet", "haiku poet", "essayist", "western painter", "buddhist", "mountaineer",
    "organic farmer", "businessman", "social entrepreneur", "politician",
    "medical politician", "social activist", "nuclear free activist", "medical critic",
    "archeology", "physical education",
    "doctor", "army doctor", "nurse",
}
MAJOR_ALIAS = {
    "orthopedics": "orthopedic surgery",
    "public health science": "public health",
    "medical school education": "medical education",
    "allergy": "allergology",
    "clinical laboratory": "laboratory medicine",
    "clinical laboratory medicine": "laboratory medicine",
}


def major_list_en(text):
    """Specialties used for edge generation: strip parentheses, split, lower-case,
    apply aliases, drop non-specialty attributes."""
    if not text or str(text).strip().lower() in ("unknown", "-", ""):
        return []
    t = re.sub(r"\s*[\(（][^)）]*[\)）]", "", str(text))
    t = re.sub(r"[\(（][^)）]*$", "", t)
    out = []
    for p in re.split(r"[,、・/;]", t):
        p = p.strip().strip(")）").strip().lower()
        if not p or p in ("unknown", "-"):
            continue
        p = MAJOR_ALIAS.get(p, p)
        if p in MAJOR_NON_SPECIALTY:
            continue
        out.append(p)
    return list(dict.fromkeys(out))


# ----------------------------------------------------------------------------
# Multilevel Louvain (self-contained, seeded)
# ----------------------------------------------------------------------------
def louvain(nodes, edges, resolution=1.0, seed=SEED, passes=50):
    rnd = random.Random(seed)
    adj = defaultdict(dict)
    for a, b in edges:
        adj[a][b] = adj[a].get(b, 0.0) + 1.0
        adj[b][a] = adj[b].get(a, 0.0) + 1.0
    for n in nodes:
        adj.setdefault(n, {})
    mapping = {n: n for n in nodes}
    while True:
        m2 = sum(sum(d.values()) for d in adj.values())
        if m2 == 0:
            break
        comm = {n: n for n in adj}
        k = {n: sum(adj[n].values()) for n in adj}
        ctot = dict(k)
        improved_any = False
        for _ in range(passes):
            improved = False
            order = list(adj)
            rnd.shuffle(order)
            for n in order:
                cn = comm[n]
                ctot[cn] -= k[n]
                nb = defaultdict(float)
                for nb_n, w in adj[n].items():
                    if nb_n != n:
                        nb[comm[nb_n]] += w
                best, bestgain = cn, nb.get(cn, 0.0) - resolution * ctot.get(cn, 0.0) * k[n] / m2
                for c, w in nb.items():
                    g = w - resolution * ctot.get(c, 0.0) * k[n] / m2
                    if g > bestgain + 1e-12:
                        bestgain, best = g, c
                comm[n] = best
                ctot[best] = ctot.get(best, 0.0) + k[n]
                if best != cn:
                    improved = improved_any = True
            if not improved:
                break
        for orig in mapping:
            mapping[orig] = comm[mapping[orig]]
        if not improved_any:
            break
        newadj = defaultdict(dict)
        for a in adj:
            ca = comm[a]
            for b, w in adj[a].items():
                cb = comm[b]
                newadj[ca][cb] = newadj[ca].get(cb, 0.0) + w
        if len(newadj) == len(adj):
            break
        adj = newadj
    return mapping


def modularity(edges, part, resolution=1.0):
    m = float(len(edges))
    if m == 0:
        return 0.0
    deg, inw, dsum = Counter(), Counter(), Counter()
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
        if part[a] == part[b]:
            inw[part[a]] += 1
    for n, d in deg.items():
        dsum[part[n]] += d
    return sum(inw[c] / m - resolution * (dsum[c] / (2 * m)) ** 2 for c in set(part.values()))


def assortativity(nodes, edges, label):
    """Newman categorical assortativity coefficient."""
    cats = sorted({label[n] for n in nodes if label.get(n)})
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    e = [[0.0] * k for _ in range(k)]
    tot = 0
    for a, b in edges:
        if label.get(a) and label.get(b):
            i, j = idx[label[a]], idx[label[b]]
            e[i][j] += 1
            e[j][i] += 1
            tot += 2
    if tot == 0:
        return float("nan")
    for i in range(k):
        for j in range(k):
            e[i][j] /= tot
    tr = sum(e[i][i] for i in range(k))
    a_i = [sum(e[i]) for i in range(k)]
    s = sum(x * x for x in a_i)
    return (tr - s) / (1 - s)


def same_school_share(edges, school):
    hit = sum(1 for a, b in edges if school.get(a) and school.get(b) and school[a] == school[b])
    return hit / float(len(edges))


def permutation_test(nodes, edges, school, blocks=None, seed=SEED, n=N_PERM):
    """Label permutation. If `blocks` is given, labels are shuffled only within
    each block (conditional permutation)."""
    obs = same_school_share(edges, school)
    rnd = random.Random(seed)
    if blocks is None:
        labels = [school[x] for x in nodes]
        draws = []
        ge = 0
        for _ in range(n):
            perm = labels[:]
            rnd.shuffle(perm)
            m = {x: perm[i] for i, x in enumerate(nodes)}
            v = same_school_share(edges, m)
            draws.append(v)
            if v >= obs:
                ge += 1
    else:
        grouped = defaultdict(list)
        for x in nodes:
            grouped[blocks[x]].append(x)
        draws = []
        ge = 0
        for _ in range(n):
            m = {}
            for _b, members in grouped.items():
                lab = [school[x] for x in members]
                rnd.shuffle(lab)
                for i, x in enumerate(members):
                    m[x] = lab[i]
            v = same_school_share(edges, m)
            draws.append(v)
            if v >= obs:
                ge += 1
    mu = sum(draws) / len(draws)
    sd = math.sqrt(sum((d - mu) ** 2 for d in draws) / len(draws))
    z = (obs - mu) / sd if sd else float("nan")
    p = (ge + 1) / float(n + 1)
    return obs, mu, z, ge, p


def community_specialty_concentration(part, nodes, main_specialty):
    """Sum of each community's modal primary specialty divided by total members."""
    grouped = defaultdict(list)
    for x in nodes:
        grouped[part[x]].append(x)
    num = tot = 0
    for _c, members in grouped.items():
        cnt = Counter(main_specialty[x] for x in members if main_specialty.get(x))
        if not cnt:
            continue
        num += cnt.most_common(1)[0][1]
        tot += sum(cnt.values())
    return 100.0 * num / tot if tot else float("nan")


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
RESULTS = []


def check(label, got, expected, tol=0.0, note=""):
    if isinstance(expected, str):
        ok = str(got) == expected
    else:
        ok = abs(float(got) - float(expected)) <= tol
    RESULTS.append(ok)
    g = f"{got:.4g}" if isinstance(got, float) else str(got)
    e = f"{expected:.4g}" if isinstance(expected, float) else str(expected)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<44s} computed={g:<10s} paper={e:<10s} {note}")


def main():
    npath = _find("nodes.csv")
    epath = _find("edges.csv")
    for p in (npath, epath):
        if not os.path.exists(p):
            raise SystemExit(
                f"[stop] not found: {os.path.basename(p)}\n"
                f"       Expected in {HERE} or {os.path.join(HERE, 'data')}."
            )

    with open(npath, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with open(epath, encoding="utf-8") as fh:
        erows = list(csv.DictReader(fh))

    nodes = [r["node_id"] for r in rows]
    school = {r["node_id"]: (r["school"] or "").strip() for r in rows}
    unit731 = {r["node_id"]: r["unit731"] == "1" for r in rows}
    community = {r["node_id"]: r["community"] for r in rows}
    specialties = {r["node_id"]: major_list_en(r["major"]) for r in rows}
    main_spec = {k: (v[0] if v else "") for k, v in specialties.items()}

    edges = [(r["source"], r["target"]) for r in erows]
    etype = Counter(r["type"] for r in erows)
    n, e = len(nodes), len(edges)

    deg = Counter()
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    degvals = sorted(deg[x] for x in nodes)

    print("=" * 88)
    print(" Verifying the article's figures against data/nodes.csv + data/edges.csv")
    print(f" seed={SEED}   permutations={N_PERM}")
    print("=" * 88)

    print("\n-- Network size --")
    check("nodes (analytic network)", n, 444)
    check("edges (undirected)", e, 8024)
    check("density", 2.0 * e / (n * (n - 1)), 0.0816, 0.0002)

    print("\n-- Edge-type decomposition --")
    check("specialty only (MAJ)", etype["MAJ"], 7236)
    check("unit only (UNIT)", etype["UNIT"], 710)
    check("both channels (BOTH)", etype["BOTH"], 78)
    check("specialty-related (MAJ+BOTH)", etype["MAJ"] + etype["BOTH"], 7314)
    check("unit-related (UNIT+BOTH)", etype["UNIT"] + etype["BOTH"], 788)
    check("specialty-related share (%)",
          100.0 * (etype["MAJ"] + etype["BOTH"]) / e, 91.2, 0.1)

    print("\n-- Alma mater --")
    csch = Counter(school[x] for x in nodes)
    todai = csch.get("Tokyo Imperial University", 0)
    check("Tokyo Imperial University", todai, 122)
    check("Tokyo Imperial share (%)", 100.0 * todai / n, 27.5, 0.1)
    check("Kyoto Imperial University", csch.get("Kyoto Imperial University", 0), 56)
    check("Keio University", csch.get("Keio University", 0), 37)
    check("Kyushu Imperial University", csch.get("Kyushu Imperial University", 0), 31)
    check("Hokkaido Imperial University", csch.get("Hokkaido Imperial University", 0), 21)

    print("\n-- School homophily: unrestricted permutation --")
    obs, mu, z, ge, p = permutation_test(nodes, edges, school)
    check("observed same-school edges (%)", 100 * obs, 13.6, 0.1)
    check("null mean (%)", 100 * mu, 11.1, 0.15)
    check("z", z, 2.83, 0.10)
    check("draws >= observed", ge, 31, 6, "sampling noise")
    check("corrected permutation p", p, 0.0064, 0.0015)
    check("edges crossing school boundaries (%)", 100 - 100 * obs, 86.4, 0.1)

    print("\n-- School homophily: permutation conditional on primary specialty --")
    obs2, mu2, z2, ge2, p2 = permutation_test(nodes, edges, school, blocks=main_spec)
    check("conditional null mean (%)", 100 * mu2, 13.0, 0.15)
    check("conditional z", z2, 2.33, 0.12)
    check("conditional p", p2, 0.019, 0.005)

    print("\n-- Assortativity (Newman, categorical) --")
    check("school r", assortativity(nodes, edges, school), 0.002, 0.001)
    check("specialty r (primary)", assortativity(nodes, edges, main_spec), 0.82, 0.01)

    print("\n-- Community structure --")
    released = len(set(community.values()))
    check("communities in the released partition", released, 19)
    check("modularity Q of the released partition", modularity(edges, community), 0.63, 0.01)
    part = louvain(nodes, edges)
    check("communities, Louvain rerun", len(set(part.values())), 19, 0,
          "implementation dependent")
    check("modularity Q, Louvain rerun", modularity(edges, part), 0.63, 0.01)
    for res, exp in ((0.5, 19), (2.0, 31)):
        pr = louvain(nodes, edges, resolution=res)
        check(f"communities at resolution {res}", len(set(pr.values())), exp, 0,
              "implementation dependent")

    print("\n-- Diagnostic 1: edge-definition sensitivity --")
    unit_edges = [(r["source"], r["target"]) for r in erows if r["type"] in ("UNIT", "BOTH")]
    core = sorted({x for ed in unit_edges for x in ed}, key=nodes.index)
    upart = louvain(core, unit_edges)
    check("unit-only edges", len(unit_edges), 788)
    check("persons in the unit-only network", len(core), 103)
    check("communities in the unit-only network", len(set(upart.values())), 12,
          0, "implementation dependent")
    check("modularity Q, unit-only network", modularity(unit_edges, upart), 0.55, 0.01)
    check("specialty concentration, full projection (%)",
          community_specialty_concentration(community, nodes, main_spec), 71.7, 0.3)
    check("specialty concentration, unit-only (%)",
          community_specialty_concentration(upart, core, main_spec), 30.0, 0.5)

    print("\n-- Degree --")
    check("maximum degree", degvals[-1], 112)
    check("median degree", degvals[len(degvals) // 2], 25)
    check("mean degree", sum(degvals) / float(len(degvals)), 36.1, 0.1)

    print("\n-- Diagnostic 2: archival silence --")
    by_name = {}
    for r in rows:
        by_name[r["name"]] = r["node_id"]
    for jp, expected in (("石井 四郎", 42), ("石井四郎", 42),
                         ("北野政次", 26), ("北野 政次", 26)):
        if jp in by_name:
            check(f"degree of {jp}", deg[by_name[jp]], expected)
    check("nodes carrying a 731 / epidemic-prevention marker",
          sum(1 for x in nodes if unit731[x]), 20)

    print("\n-- Table 2: distribution of primary specialties --")
    cm = Counter()
    for x in nodes:
        for s in specialties[x]:
            cm[s] += 1
    table2 = [("internal medicine", 86), ("surgery", 50), ("bacteriology", 27),
              ("pathology", 26), ("ophthalmology", 22), ("orthopedic surgery", 22),
              ("psychiatry", 20), ("pediatrics", 19), ("hygiene", 18),
              ("anatomy", 16), ("obstetrics and gynecology", 15),
              ("otorhinolaryngology", 15)]
    for name, exp in table2:
        check(name, cm.get(name, 0), exp)

    print("\n" + "=" * 88)
    failed = RESULTS.count(False)
    print(f" {len(RESULTS) - failed} of {len(RESULTS)} checks passed"
          + ("" if failed == 0 else f"   -  {failed} FAILED"))
    if failed:
        print(" Community counts and the unit-only partition depend on the Louvain")
        print(" implementation; a mismatch there is expected with other libraries.")
        print(" A mismatch in node, edge, degree, or permutation figures is not.")
    print("=" * 88)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

import os
import glob
import re
import json
import itertools
import math
import hashlib
import time
import random
import urllib.request
import urllib.parse
import ssl
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import networkx as nx
import unicodedata
from pyvis.network import Network
from collections import defaultdict, Counter
from scipy.spatial import KDTree
import community.community_louvain as community_louvain
# ══════════════════════════════════════════════════════════════════
MAX_EDGES      = 15000
# ══════════════════════════════════════════════════════════════════
# Old-form -> new-form kanji normalization table (merges verified-list new-form with roster old-form)
NORM = {'澤':'沢','齋':'斎','齊':'斉','髙':'高','邊':'辺','邉':'辺','條':'条','瀨':'瀬','濱':'浜',
        '應':'応','櫻':'桜','圓':'円','學':'学','醫':'医','國':'国','廣':'広','德':'徳','龍':'竜',
        '瀧':'滝','增':'増','峯':'峰','﨑':'崎','眞':'真','彌':'弥','惠':'恵','榮':'栄','晉':'晋',
        '寬':'寛','黑':'黒','兒':'児','關':'関'}
def norm_key(name):
    s = re.sub(r'\s+', '', str(name)).replace('　', '')
    return ''.join(NORM.get(c, c) for c in s)
def dyn_color(name):
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    return "#{:02x}{:02x}{:02x}".format(min(255, ((h >> 16) & 0xFF) + 60), min(255, ((h >> 8)  & 0xFF) + 60), min(255, (h & 0xFF) + 60))
def get_heatmap_color(val, max_val):
    ratio = val / max_val if max_val > 0 else 0
    h = int((1.0 - ratio) * 240)
    return f"hsl({h}, 100%, 50%)"
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
def extract_all_years(text):
    years = set()
    for m in re.finditer(r'(18\d{2}|19\d{2}|20\d{2})', text): years.add(int(m.group(1)))
    for m in re.finditer(r'昭和\s*(\d+)', text): years.add(1925 + int(m.group(1)))
    for m in re.finditer(r'大正\s*(\d+)', text): years.add(1911 + int(m.group(1)))
    for m in re.finditer(r'明治\s*(\d+)', text): years.add(1867 + int(m.group(1)))
    for m in re.finditer(r'平成\s*(\d+)', text): years.add(1988 + int(m.group(1)))
    return sorted(list(years))
def get_grad_year(text):
    match = re.search(r'(18\d{2}|19\d{2}|20\d{2}|昭和\s*\d+|大正\s*\d+|明治\s*\d+|平成\s*\d+)[^\d]{0,5}卒', text)
    if match:
        y_str = match.group(1).replace(' ', '')
        if y_str.startswith('1') or y_str.startswith('2'): return int(y_str)
        elif '昭和' in y_str: return 1925 + int(re.search(r'\d+', y_str).group())
        elif '大正' in y_str: return 1911 + int(re.search(r'\d+', y_str).group())
        elif '明治' in y_str: return 1867 + int(re.search(r'\d+', y_str).group())
    return None
def normalize_school_ja(s):
    if not s or s in ("-", "Unknown"): return "Unknown"
    # 졸업연도 등 괄호 꼬리표 제거 (예: '岡山医科大学 (昭和9年卒)' -> '岡山医科大学')
    s = re.sub(r"\s*[\(（].*?[\)）]\s*", "", str(s)).strip()
    if not s: return "Unknown"
    s_clean = s.replace(" ", "").replace("　", "")
    if re.search(r"東京帝|東大|東京大", s_clean): return "東京帝国大学"
    if re.search(r"京都帝|京大|京都大", s_clean): return "京都帝国大学"
    if re.search(r"九州帝|九大|九州大", s_clean): return "九州帝国大学"
    if re.search(r"北海道帝|北大|北海道大", s_clean): return "北海道帝国大学"
    if re.search(r"東北帝|東北大", s_clean): return "東北帝国大学"
    if re.search(r"大阪帝|阪大|大阪大", s_clean): return "大阪帝国大学"
    if re.search(r"名古屋帝|名大|名古屋大", s_clean): return "名古屋帝国大学"
    if re.search(r"慶應|慶応", s_clean): return "慶應義塾大学"
    if re.search(r"千葉", s_clean): return "千葉医科大学"
    if s_clean == "帝国대" or s_clean == "帝国大学": return "東京帝国大学"
    # 표기 변이(약칭·이표기) 통합 — 같은 학교가 다른 라벨로 쪼개지는 것 방지
    _aliases = {
        "慈恵会医科大学": "慈恵医科大学", "慈恵医大": "慈恵医科大学",
        "台北医科専門学校": "台北医学専門学校",
        "昭和医専": "昭和医学専門学校", "長崎医専": "長崎医科大学",
        "東京医科専門学校": "東京医学専門学校", "日本医学校": "日本医科大学",
        "新潟医学専門学校": "新潟医科大学", "熊本医学専門学校": "熊本医科大学",
        "京都府立医学専門学校": "京都府立医科大学", "金沢医科専門学校": "金沢医科大学",
    }
    return _aliases.get(s_clean, s)
def normalize_major(s):
    """전공명 정규화: '陸軍軍医(細菌学)'→'細菌学', '内科(結核病学)'→'内科' 등 표기 변이 통합.
    - '陸軍軍医/海軍軍医(X)' 형태는 괄호 안 전공 X를 사용(군의 접두는 전공이 아님).
    - 그 외 'Y(세부)' 형태는 괄호 앞 대분류 Y로 통합.
    - 콤마/점 구분 다중 전공은 각각 정규화."""
    if not s or str(s).strip() in ("-", "Unknown", ""): return s
    parts = re.split(r'[,、]', str(s)); out = []
    for p in parts:
        p = p.strip()
        m = re.match(r'^(?:陸軍軍医|海軍軍医|軍医)\s*[\(（](.+?)[\)）]\s*$', p)
        if m:
            p = m.group(1).strip()
        else:
            p = re.sub(r'\s*[\(（].*?[\)）]', '', p).strip()
        p = re.sub(r'^(?:陸軍軍医|海軍軍医|軍医)$', '', p).strip()
        if p: out.append(p)
    seen = list(dict.fromkeys(out))
    return ", ".join(seen) if seen else s
def extract_real_units(text):
    """근무지/소속에서 '실제 배속 부대'만 추출·정규화한다.
    직함(陸軍軍医)·계급(軍医少尉)·일반 서술(軍務·兵役·中国戦線 등)은 부대가 아니므로 제외.
    표기 변이는 통합: 731=관동군방역급수부, 軍医学校=陸軍軍医学校 등."""
    t = str(text); out = []
    def add(x):
        if x and x not in out: out.append(x)
    if re.search(r'満州第?\s*731\s*部隊|関東軍防疫給水部', t): add('満州第731部隊')
    if re.search(r'中支(?:那)?防疫給水部', t): add('中支防疫給水部')
    if re.search(r'北支(?:那)?防疫給水部', t): add('北支防疫給水部')
    if re.search(r'南方軍?防疫給水部', t): add('南方軍防疫給水部')
    if re.search(r'(?:陸軍)?軍医学校|防疫研究室', t): add('陸軍軍医学校')
    for m in re.finditer(r'(\d+)\s*師団', t): add('第%s師団' % m.group(1))
    if re.search(r'近衛師団', t): add('近衛師団')
    for m in re.finditer(r'(\d+)\s*連隊', t): add('第%s連隊' % m.group(1))
    for m in re.finditer(r'第\s*(\d+)\s*軍(?!医)', t): add('第%s軍' % m.group(1))
    for m in re.finditer(r'[一-龥]{2,8}病院', t): add(m.group(0))
    for m in re.finditer(r'[一-龥]{1,4}要塞', t): add(m.group(0))
    for m in re.finditer(r'[一-龥]{1,4}鎮守府', t): add(m.group(0))
    return out
def split_affiliations(unit_text):
    if not unit_text or unit_text in ("-", "Unknown", "―"): return "Unknown", "Unknown"
    mil = extract_real_units(unit_text)
    gen_parts = [p.strip() for p in re.split(r'[,、・\s]+|<br\s*/?>', str(unit_text)) if p.strip()
                 and not re.search(r'軍|兵|隊|師団|連隊|防疫|病院|衛生|要塞|艦隊|鎮守府|戦|俘虜', p)]
    return (", ".join(gen_parts) if gen_parts else "Unknown"), (", ".join(mil) if mil else "Unknown")
GROUP_COLORS = {
    "army": "#C0504D",
    "navy": "#4472C4",
    "mil_med": "#E0A000",
    "mil_admin": "#7B5EA7",
    "unknown_branch": "#8C8C8C",
    "civilian": "#555555"
}
FIELD_ADJ_COLOR = "#D9923E"  # Field-adjacent (indirect candidate) color
def is_field_adjacent(major_ja, post, is_war):
    """Civilian (no military service) whose specialty/affiliation is military-medicine-adjacent (bacteriology, epidemic prevention, hygiene, colonial medicine)."""
    if is_war: return False
    t = str(major_ja) + " " + str(post)
    fld = bool(re.search(r'細菌|防疫|衛生|血清|伝染病|熱帯|寄生虫|ウイルス|ワクチン|免疫|微生物', t))
    inst = bool(re.search(r'台北|京城|セブランス|満州|満洲|奉天|大連|関東|同仁会|伝染病研究所|北里|伝研', str(post)))
    return fld or inst
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xlsx_files = glob.glob(os.path.join(script_dir, "*.xlsx"))
    # Process the verified list last (roster first -> verified list overrides war/group)
    xlsx_files.sort(key=lambda x: 1 if ("검증본" in unicodedata.normalize('NFC', x) or "군의학" in unicodedata.normalize('NFC', x)) else 0)
    if not xlsx_files:
        print("No Excel (.xlsx) files found.")
        return
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    cache_file = os.path.join(script_dir, "translation_cache.json")
    cache = json.load(open(cache_file, encoding="utf-8")) if os.path.exists(cache_file) else {}
    MANUAL_TRANS = {
        "東京帝国大学": "Tokyo Imperial University",
        "京都帝国大学": "Kyoto Imperial University",
        "陸軍軍医学校": "Army Medical College",
        "北野政次": "Masaji Kitano",
        "石井四郎": "Shiro Ishii",
        "石川太刀雄": "Tachio Ishikawa"
    }
    def save_cache():
        json.dump(cache, open(cache_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    def translate_one(text):
        text = str(text).strip()
        if not text or text in ("-", "―", "Unknown"): return "Unknown"
        if text in MANUAL_TRANS: return MANUAL_TRANS[text]
        for k, v in MANUAL_TRANS.items():
            if k in text: text = text.replace(k, v)
        if text in cache: return cache[text]
        url = ("https://translate.googleapis.com/translate_a/single"
               "?client=gtx&sl=ja&tl=en&dt=t&q=" + urllib.parse.quote(text))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            result = json.loads(urllib.request.urlopen(req, context=ctx, timeout=5).read())
            t = "".join(s[0] for s in result[0])
            cache[text] = t; time.sleep(0.05); return t
        except Exception:
            cache[text] = text; return text
    def translate_unique(texts, label=""):
        unique  = {str(t).strip() for t in texts if t and str(t).strip() not in ("", "-", "―", "Unknown")}
        new     = [t for t in unique if t not in cache and t not in MANUAL_TRANS]
        if new: print(f"  [{label}] {len(unique)} unique, {len(new)} to translate")
        for i, t in enumerate(new):
            translate_one(t)
            if (i + 1) % 50 == 0: save_cache(); print(f"    ... {i+1}/{len(new)}")
        save_cache()
        return {t: translate_one(t) for t in unique}
    def clean_text(text):
        return re.sub(r'<br\s*/?>|</br>', ' ', str(text)).strip()
    print(f"Loading Excel files (Perfect Merging to prevent data loss)...")
    raw_dict = {}
    for file in xlsx_files:
        norm_file = unicodedata.normalize('NFC', file)
        is_verified_file = "검증본" in norm_file or "군의학" in norm_file
        try:
            df = pd.read_excel(file).fillna("")
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                nv = clean_text(row_dict.get("名前 (ふりがな)", "") or row_dict.get("이름", "") or row_dict.get("名前", ""))
                if not nv: continue
                col_furi = clean_text(row_dict.get("읽기", "") or row_dict.get("ふりがな", ""))
                col_unit = clean_text(row_dict.get("근무지/소속", "") or row_dict.get("勤務先・役職", ""))
                col_major = normalize_major(clean_text(row_dict.get("분야", "") or row_dict.get("分野", "-")))
                col_rank = clean_text(row_dict.get("계급", "") or row_dict.get("階級", "-"))
                col_life = clean_text(row_dict.get("생몰", "") or row_dict.get("生没年", "") or row_dict.get("出生~死亡", "") or row_dict.get("出生～死亡", "") or "-")
                col_school = clean_text(row_dict.get("출신학교", "") or row_dict.get("出身学校", "-"))
                col_cat = clean_text(row_dict.get("분류", "") or row_dict.get("軍医官経歴", ""))
                col_notes = clean_text(row_dict.get("비고", "") or row_dict.get("備考", ""))
                abroad_ja = clean_text(row_dict.get("유학", "") or row_dict.get("留学", ""))
                if col_furi:
                    kanji_name = re.sub(r"[\(（].*?[\)）]", "", nv).replace(" ", " ").strip()
                    trans_target = col_furi
                else:
                    match = re.search(r"[\(（](.*?)[\)）]", nv)
                    if match:
                        trans_target = match.group(1).strip()
                        kanji_name = re.sub(r"[\(（].*?[\)）]", "", nv).replace(" ", " ").strip()
                    else:
                        kanji_name = nv.replace(" ", " ").strip()
                        trans_target = kanji_name
                if not kanji_name: continue
                # Merge by normalized-kanji key (unify the same person across verified list and roster)
                merge_key = norm_key(kanji_name)
                cat_str = col_cat + col_unit + col_notes
                # Trust only verified-list membership for military classification -> blocks re-entry of nurse/non-physician keywords
                is_war_curr = is_verified_file and not bool(re.search(r"해군|海軍", cat_str))  # exclude Navy
                is_731_curr = bool(re.search(r'731|防疫給水部|石井|방역급수|이시이', cat_str))
                purged_curr = bool(re.search(r'追放', cat_str))           # postwar public-office / teaching purge
                _byrs = extract_all_years(col_life)
                birth_curr = min(_byrs) if _byrs else None               # birth year (cohort)
                group_curr = "civilian"
                if is_war_curr:
                    if re.search(r"육군|陸軍", cat_str): group_curr = "army"
                    elif re.search(r"해군|海軍", cat_str): group_curr = "navy"
                    elif re.search(r"군의학|군의관|기술관|軍医", cat_str): group_curr = "mil_med"
                    elif re.search(r"의무행정|행정|行政", cat_str): group_curr = "mil_admin"
                    else: group_curr = "unknown_branch"
                raw_school = normalize_school_ja(col_school)
                gen_ja, mil_ja = split_affiliations(col_unit)
                if merge_key not in raw_dict:
                    raw_dict[merge_key] = {
                        "name_ja": kanji_name,
                        "name_trans": trans_target,
                        "school_ja": raw_school,
                        "major_ja": col_major,
                        "post": col_unit,
                        "gen_unit_ja": gen_ja,
                        "mil_unit_ja": mil_ja,
                        "rank": col_rank,
                        "life": col_life,
                        "abroad_ja": abroad_ja,
                        "is_war": is_war_curr,
                        "unit731": is_731_curr,
                        "purged": purged_curr,
                        "birth": birth_curr,
                        "group": group_curr
                    }
                else:
                    ex = raw_dict[merge_key]
                    ex["is_war"] = ex["is_war"] or is_war_curr
                    ex["unit731"] = ex["unit731"] or is_731_curr
                    ex["purged"] = ex.get("purged", False) or purged_curr
                    if ex.get("birth") is None and birth_curr is not None: ex["birth"] = birth_curr
                    if ex["school_ja"] in ("-", "Unknown", "") and raw_school not in ("-", "Unknown", ""):
                        ex["school_ja"] = raw_school
                    if ex["major_ja"] in ("-", "") and col_major not in ("-", ""):
                        ex["major_ja"] = col_major
                    if ex["post"] in ("-", "") and col_unit not in ("-", ""):
                        ex["post"] = col_unit
                        ex["gen_unit_ja"] = gen_ja
                        ex["mil_unit_ja"] = mil_ja
                    if ex["rank"] in ("-", "") and col_rank not in ("-", ""):
                        ex["rank"] = col_rank
                    if ex["group"] == "civilian" and group_curr != "civilian":
                        ex["group"] = group_curr
                    if len(kanji_name) > len(ex["name_ja"]):
                        ex["name_ja"] = kanji_name
        except Exception as e:
            pass
    raw = list(raw_dict.values())
    print(f"Merged & Extracted {len(raw)} total records safely.")
    name_map     = translate_unique([r["name_trans"] for r in raw], "Names")
    school_map   = translate_unique([r["school_ja"] for r in raw], "Schools")
    major_map    = translate_unique([r["major_ja"] for r in raw], "Majors")
    gen_unit_map = translate_unique([r["gen_unit_ja"] for r in raw if r["gen_unit_ja"] != "Unknown"], "GenUnits")
    mil_unit_map = translate_unique([r["mil_unit_ja"] for r in raw if r["mil_unit_ja"] != "Unknown"], "MilUnits")
    def get_list(text):
        if not text or text.lower() in ("unknown", "-", ""): return []
        return [w.strip() for w in text.replace("、", ",").replace("・", ",").split(",") if w.strip()]
    persons = []
    war_count = sum(1 for r in raw if r["is_war"])
    for i, r in enumerate(raw):
        p_mil_en = mil_unit_map.get(r["mil_unit_ja"], "Unknown") if r["mil_unit_ja"] != "Unknown" else "Unknown"
        p_gen_en = gen_unit_map.get(r["gen_unit_ja"], "Unknown") if r["gen_unit_ja"] != "Unknown" else "Unknown"
        translated_name = name_map.get(r["name_trans"], r["name_ja"])
        if translated_name != r["name_ja"]: translated_name = translated_name.title()
        persons.append({
            "uid": f"n{i}",
            "name": translated_name,
            "name_ja": r["name_ja"],
            "school": school_map.get(r["school_ja"], "Unknown") or "Unknown",
            "school_ja": r["school_ja"],
            "major": major_map.get(r["major_ja"], "Unknown") or "Unknown",
            "major_ja": r["major_ja"],
            "gen_unit": p_gen_en,
            "gen_unit_ja": r["gen_unit_ja"],
            "mil_unit": p_mil_en,
            "mil_unit_ja": r["mil_unit_ja"],
            "post": r["post"],
            "rank": r["rank"],
            "life": r["life"],
            "group": r["group"],
            # 시계열 정제(진단③): 1925년 이후 출생자는 종전(1945) 시 20세 이하로 전시 군의가 불가하므로 분석 모집단에서 제외
            "is_war": (r["is_war"] and not (r.get("birth") is not None and r["birth"] >= 1925)),
            "unit731": r["unit731"],
            "purged": r.get("purged", False),
            "birth": r.get("birth"),
            "field_adj": is_field_adjacent(r["major_ja"], r["post"], r["is_war"]),
            "color_group": GROUP_COLORS.get(r["group"], "#555555"),
            "is_elite": bool(re.search(r"あり|欧|米|独|仏|英|留|渡", r["abroad_ja"]))
        })
    mil_idx, major_idx, compound_idx = defaultdict(list), defaultdict(list), defaultdict(list)
    for p in persons:
        if p["is_war"]:
            for u in get_list(p["mil_unit"]): mil_idx[u].append(p["uid"])
            for m in get_list(p["major"]): major_idx[m].append(p["uid"])
            for u in get_list(p["mil_unit"]):
                for m in get_list(p["major"]): compound_idx[(u, m)].append(p["uid"])
    print(f"Calculating True Network Statistics...")
    true_neighbors_info = defaultdict(dict)
    TrueG = nx.Graph()
    for p in persons:
        if p["is_war"]: TrueG.add_node(p["uid"])
    for u_name, members in mil_idx.items():
        if len(members) > 1:
            for a, b in itertools.combinations(members, 2):
                true_neighbors_info[a][b] = {"type": "Unit", "detail": f"Unit: {u_name}"}
                true_neighbors_info[b][a] = {"type": "Unit", "detail": f"Unit: {u_name}"}
                TrueG.add_edge(a, b, weight=1)
    for m_name, members in major_idx.items():
        if len(members) > 1:
            for a, b in itertools.combinations(members, 2):
                if b in true_neighbors_info[a]:
                    true_neighbors_info[a][b] = {"type": "Both", "detail": f"Major & Unit Match"}
                    TrueG.add_edge(a, b, weight=2)
                else:
                    true_neighbors_info[a][b] = {"type": "Major", "detail": f"Major: {m_name}"}
                    TrueG.add_edge(a, b, weight=1)
                if a in true_neighbors_info[b]:
                    true_neighbors_info[b][a] = {"type": "Both", "detail": f"Major & Unit Match"}
                    TrueG.add_edge(a, b, weight=2)
                else:
                    true_neighbors_info[b][a] = {"type": "Major", "detail": f"Major: {m_name}"}
                    TrueG.add_edge(a, b, weight=1)
    for (u_name, m_name), members in compound_idx.items():
        if len(members) > 1:
            for a, b in itertools.combinations(members, 2):
                true_neighbors_info[a][b] = {"type": "Both", "detail": f"Both: {u_name} & {m_name}"}
                true_neighbors_info[b][a] = {"type": "Both", "detail": f"Both: {u_name} & {m_name}"}
    true_score = {uid: len(nbs) for uid, nbs in true_neighbors_info.items()}
    isolated_true = [u for u in TrueG.nodes() if TrueG.degree(u) == 0]
    TrueG.remove_nodes_from(isolated_true)
    print(f"Running Louvain Community Detection (Modularity)...")
    random.seed(42)  # 커뮤니티 탐지 재현성 고정 (모듈성 Q가 매 실행 동일하게 산출되도록)
    partition = community_louvain.best_partition(TrueG, weight='weight', resolution=1.0) if TrueG.number_of_nodes() else {}
    unique_comms = set(partition.values())
    comm_colors = {c: f"hsl({int((c * 137.5) % 360)}, 85%, 60%)" for c in unique_comms}
    comm_majors = defaultdict(list)
    for uid, c_id in partition.items():
        p = next((x for x in persons if x['uid']==uid), None)
        if p and p['major'] != 'Unknown':
            comm_majors[c_id].extend([m.strip() for m in p['major'].split(',')])
    comm_dom_major = {}
    for c_id, m_list in comm_majors.items():
        if m_list: comm_dom_major[c_id] = Counter(m_list).most_common(1)[0][0]
        else: comm_dom_major[c_id] = "Mixed/General"
    print(f"Calculating Betweenness Centrality...")
    k_samples = min(len(TrueG.nodes()), 500) if TrueG.number_of_nodes() else 0
    betweenness = nx.betweenness_centrality(TrueG, k=k_samples, weight='weight', seed=42) if k_samples else {}
    max_bet = max(betweenness.values()) if betweenness else 1
    sorted_bet = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    bet_ranks = {uid: rank+1 for rank, (uid, score) in enumerate(sorted_bet)}
    # ══════════════════════════════════════════════════════════════════
    # Real statistics (replacing the fake placeholders) -> injected into the modal
    # ══════════════════════════════════════════════════════════════════
    print(f"Computing real statistics (permutation / modularity / brokerage)...")
    import statistics as _st
    pmap = {p["uid"]: p for p in persons}
    war_nodes = list(TrueG.nodes())
    edges_tg = list(TrueG.edges())
    # (1) School homophily permutation test
    sch = {u: pmap[u]["school"] for u in war_nodes}
    def _same_sch(mp):
        if not edges_tg: return 0.0
        c = sum(1 for a, b in edges_tg if mp[a] == mp[b] and mp[a] not in ("Unknown", "", None))
        return c / len(edges_tg)
    obs_h = _same_sch(sch)
    _labs = [sch[u] for u in war_nodes]
    random.seed(42)  # 순열검정 재현성 고정 (z·obs가 매 실행 동일하게 산출되도록)
    _nulls = []
    for _ in range(500):
        _sh = _labs[:]; random.shuffle(_sh)
        _nulls.append(_same_sch(dict(zip(war_nodes, _sh))))
    null_mean = _st.mean(_nulls) if _nulls else 0.0
    null_sd = (_st.pstdev(_nulls) if len(_nulls) > 1 else 0.0) or 1e-9
    z_homophily = (obs_h - null_mean) / null_sd
    # (2) Modularity: full (unit+major) vs specialty(major)-edge-removed (robustness)
    try:
        mod_full = community_louvain.modularity(partition, TrueG, weight='weight') if edges_tg else 0.0
    except Exception:
        mod_full = 0.0
    TrueG_nm = nx.Graph(); TrueG_nm.add_nodes_from(war_nodes)
    for u_name, members in mil_idx.items():
        if len(members) > 1:
            for a, b in itertools.combinations(members, 2): TrueG_nm.add_edge(a, b)
    TrueG_nm.remove_nodes_from([n for n in list(TrueG_nm.nodes()) if TrueG_nm.degree(n) == 0])
    if TrueG_nm.number_of_edges() > 0:
        random.seed(42)  # 전공엣지 제거 모듈성 재현성 고정
        _pnm = community_louvain.best_partition(TrueG_nm, resolution=1.0)
        mod_nomajor = community_louvain.modularity(_pnm, TrueG_nm)
    else:
        mod_nomajor = 0.0
    # ── 본문 재작성용 추가 지표 (콘솔 출력) ──
    _n2u = {p.get("name_ja"): p["uid"] for p in persons if p.get("is_war")}
    for _w in ("石井四郎", "北野政次"):
        _u = _n2u.get(_w)
        _dg = TrueG.degree(_u) if (_u is not None and _u in TrueG) else "분석망 외"
        print(f"   [추가] {_w} degree={_dg}")
    from collections import Counter as _Ctr
    _topd = sorted(((TrueG.degree(u), pmap[u]["school_ja"]) for u in war_nodes), reverse=True)[:10]
    print("   [추가] 연결도 상위10 출신교:", dict(_Ctr(s for _, s in _topd)))
    print(f"   [추가] 부대(전공엣지 제거 후) 엣지수={TrueG_nm.number_of_edges()}")
    print(f"   [추가] 학벌 경계 넘는 연결={100*(1-obs_h):.1f}%")
    # (3) Broker (top-10 betweenness) structure analysis
    top10 = [u for u, _ in sorted(betweenness.items(), key=lambda x: -x[1])[:10]]
    _degd = dict(TrueG.degree())
    _topdeg = set([u for u, _ in sorted(_degd.items(), key=lambda x: -x[1])[:10]])
    brk_overlap = len(set(top10) & _topdeg)
    brk_dual = sum(1 for u in top10 if "," in (pmap[u]["major"] or ""))
    brk_todai = sum(1 for u in top10 if ("Tokyo" in (pmap[u]["school"] or "") or "東京" in (pmap[u]["school_ja"] or "")))
    # (4) Postwar teaching/public-office purge
    purged_total = sum(1 for p in persons if p.get("purged"))
    purged_war = sum(1 for u in war_nodes if pmap[u].get("purged"))
    purged_brk = sum(1 for u in top10 if pmap[u].get("purged"))
    purged_war_pct = (purged_war * 100.0 / len(war_nodes)) if war_nodes else 0.0
    # (5) Density
    _nw = len(war_nodes)
    war_density = (2.0 * len(edges_tg)) / (_nw * (_nw - 1)) if _nw > 1 else 0.0
    print(f"   homophily obs={obs_h:.3f} null={null_mean:.3f} z={z_homophily:.2f} | "
          f"modularity {mod_full:.3f}→{mod_nomajor:.3f} | broker overlap={brk_overlap}/10 dual={brk_dual} todai={brk_todai} | "
          f"purged war={purged_war}({purged_war_pct:.1f}%) brokers={purged_brk}/10 | density={war_density:.4f}")

    # ===== figures for the manuscript =====
    from collections import Counter as _C
    _sch=_C(pmap[u]["school"] for u in war_nodes)
    _maj=_C(m.strip() for u in war_nodes for m in str(pmap[u].get("major_ja") or pmap[u].get("major") or "").replace("、",",").replace("・",",").split(",") if m.strip())
    _degvals=sorted(dict(TrueG.degree()).values()) or [0]
    _todai=sum(v for k,v in _sch.items() if ("Tokyo" in str(k)) or ("東京" in str(k)))
    print("\n========== figures for the manuscript ==========")
    print(f"N(army nodes)={len(war_nodes)}  E(edges)={len(edges_tg)}  density={war_density:.4f}")
    print(f"Homophily: observed={obs_h*100:.1f}%  null={null_mean*100:.1f}%  z={z_homophily:.2f}")
    print(f"Modularity (full -> specialty-edges removed): {mod_full:.3f} -> {mod_nomajor:.3f}")
    print(f"degree  max={_degvals[-1]}  median={_degvals[len(_degvals)//2]}  mean={sum(_degvals)/len(_degvals):.1f}")
    print(f"Tokyo Imperial={_todai} ({_todai*100.0/len(war_nodes):.1f}%)")
    print("School top10:", _sch.most_common(10))
    print("Specialty top12 (Table 1):", _maj.most_common(12))
    print("===========================================================\n")

    def cohort_color(b):
        if not b: return "#555555"
        idx = {1850: 0, 1860: 1, 1870: 2, 1880: 3, 1890: 4, 1900: 5, 1910: 6, 1920: 7}.get((b // 10) * 10, 8)
        return f"hsl({max(0, 210 - idx * 26)}, 70%, 55%)"
    print(f"️ Building Visually Compressed Graph...")
    G = nx.Graph()
    for p in persons: G.add_node(p["uid"])
    for (u_name, m_name), members in compound_idx.items():
        if len(members) > 1:
            for m in members:
                targets = random.sample(members, min(10, len(members)-1))
                for t in targets:
                    if m != t: G.add_edge(m, t, etype="both", title=f"Shared: {u_name} & {m_name}")
    for u_name, members in mil_idx.items():
        if len(members) > 1:
            for m in members:
                targets = random.sample(members, min(10, len(members)-1))
                for t in targets:
                    if m != t: G.add_edge(m, t, etype="unit", title=f"Shared Unit: {u_name}")
    for m_name, members in major_idx.items():
        if len(members) > 1:
            for m in members:
                targets = random.sample(members, min(10, len(members)-1))
                for t in targets:
                    if m != t: G.add_edge(m, t, etype="major", title=f"Shared Major: {m_name}")
    connected = set(G.nodes())
    pos = {}
    def get_geo_coords(s_ja):
        if re.search(r"北海道|札幌", s_ja): return 141.34, 43.06
        if re.search(r"東北|仙台", s_ja): return 140.87, 38.25
        if "新潟" in s_ja: return 139.04, 37.91
        if "金沢" in s_ja: return 136.65, 36.56
        if "千葉" in s_ja: return 140.10, 35.61
        if re.search(r"東京帝|東大", s_ja): return 139.76, 35.71
        if "慶應" in s_ja: return 139.74, 35.64
        if "慈恵" in s_ja: return 139.75, 35.66
        if re.search(r"日本医|日大", s_ja): return 139.75, 35.72
        if "昭和" in s_ja: return 139.70, 35.60
        if "順天堂" in s_ja: return 139.76, 35.70
        if re.search(r"東京|横浜", s_ja): return 139.70, 35.65
        if re.search(r"名古屋|愛知", s_ja): return 136.93, 35.15
        if re.search(r"京都帝|京大", s_ja): return 135.78, 35.02   # regex match for school-name variants
        if re.search(r"大阪帝|阪大", s_ja): return 135.52, 34.82   # regex match for school-name variants
        if re.search(r"大阪|関西", s_ja): return 135.50, 34.69
        if re.search(r"神戸|兵庫", s_ja): return 135.19, 34.69
        if "岡山" in s_ja: return 133.91, 34.66
        if "広島" in s_ja: return 132.45, 34.38
        if "山口" in s_ja: return 131.40, 34.10
        if "徳島" in s_ja: return 134.50, 34.00
        if re.search(r"九州|福岡", s_ja): return 130.42, 33.62
        if "久留米" in s_ja: return 130.51, 33.31
        if "熊本" in s_ja: return 130.73, 32.81
        if "長崎" in s_ja: return 129.86, 32.77
        if "鹿児島" in s_ja: return 130.50, 31.50
        if re.search(r"京城|セブランス|ソウル", s_ja): return 126.97, 37.56
        if "台北" in s_ja: return 121.53, 25.01
        if re.search(r"満州|奉天|大連", s_ja): return 123.43, 41.80
        return 137.0, 37.0
    # [edit] Keep the Japan-map shape but increase inter-city spacing (SCALE_MAP) -> clearer separation
    SCALE_MAP = 24000
    CENTER_LON = 136.0; CENTER_LAT = 36.0
    sch_groups = defaultdict(list)
    for p in persons:
        if p["uid"] in connected: sch_groups[p["school_ja"]].append(p)
    region_schools = defaultdict(list)
    for s_ja, scholars in sch_groups.items():
        lon, lat = get_geo_coords(s_ja)
        reg_key = f"{lon}_{lat}"
        region_schools[reg_key].append((s_ja, scholars, lon, lat))
    for reg_key, schools in region_schools.items():
        schools.sort(key=lambda x: len(x[1]), reverse=True)
        lon, lat = schools[0][2], schools[0][3]
        base_x = (lon - CENTER_LON) * SCALE_MAP
        base_y = -(lat - CENTER_LAT) * SCALE_MAP
        # [edit] Tighter clusters per university (smaller radius); modest separation among universities in the same city
        center_radius = max(380, math.sqrt(len(schools[0][1])) * 165)
        for i, (s_ja, scholars, _, _) in enumerate(schools):
            R_school = max(220, math.sqrt(len(scholars)) * 165)
            if i == 0: sx, sy = base_x, base_y
            else:
                dist = center_radius + R_school + 5000 + (i * 1600)
                angle = (i - 1) * (2 * math.pi / max(1, len(schools)-1))
                sx = base_x + dist * math.cos(angle)
                sy = base_y + dist * math.sin(angle)
            random.shuffle(scholars)
            for idx, p in enumerate(scholars):
                r = R_school * math.sqrt(idx / max(1, len(scholars)))
                theta = idx * 137.508 * (math.pi / 180.0)
                pos[p["uid"]] = np.array([sx + r * math.cos(theta), sy + r * math.sin(theta)])
    print("Applying Fast KDTree physics...")
    node_uids = list(connected)
    if node_uids:
        pos_ary = np.array([pos[u] for u in node_uids], dtype=float)
        # [edit] Smaller collision padding -> clusters stay tight rather than scattering
        radii = np.array([min(8 + true_score.get(u, 0) * 0.04, 36) for u in node_uids])
        for _ in range(12):
            tree = KDTree(pos_ary)
            pairs = tree.query_pairs(np.max(radii) * 2 + 8)
            for i, j in pairs:
                dx = pos_ary[i][0] - pos_ary[j][0]
                dy = pos_ary[i][1] - pos_ary[j][1]
                dist = math.hypot(dx, dy)
                min_dist = radii[i] + radii[j] + 6
                if dist < min_dist:
                    if dist < 0.01:
                        dx, dy = random.random() - 0.5, random.random() - 0.5
                        dist = math.hypot(dx, dy)
                    overlap = (min_dist - dist) / 2
                    pos_ary[i][0] += (dx / dist) * overlap * 0.5
                    pos_ary[i][1] += (dy / dist) * overlap * 0.5
                    pos_ary[j][0] -= (dx / dist) * overlap * 0.5
                    pos_ary[j][1] -= (dy / dist) * overlap * 0.5
        for i, u in enumerate(node_uids): pos[u] = pos_ary[i]
    net = Network(height="100vh", width="100%", bgcolor="#0f172a", font_color="white", cdn_resources="remote")
    net.toggle_physics(False)
    node_meta = {}
    for p in persons:
        uid = p["uid"]
        x, y = pos.get(uid, (0, 0))
        t_score = true_score.get(uid, 0)
        b_score = betweenness.get(uid, 0.0)
        b_rank = bet_ranks.get(uid, 9999)
        is_broker = b_rank <= 20
        n_shape = "star" if is_broker else "dot"
        # [edit] Civilian node size 5 -> 9 (visible when all lit at S0); military non-brokers start at 12
        n_size = min(40 + (t_score * 0.05), 150) if is_broker else (min(12 + (t_score * 0.02), 45) if p["is_war"] else 9)
        comm_id = partition.get(uid, -1)
        dom_major_label = comm_dom_major.get(comm_id, "Mixed")
        color_comm = comm_colors.get(comm_id, "#334155")
        color_bg = dyn_color(p["school"])
        color_bet = get_heatmap_color(b_score, max_bet)
        color_grp = p["color_group"]
        b_width = 6 if is_broker else 2
        init_bg = color_bg
        init_border = color_comm if p["is_war"] else color_bg
        clean_tooltip = f"{p['name_ja']}\n"
        if p["is_war"] and p['mil_unit'] not in ("", "-", "Unknown"):
            clean_tooltip += f"Post: {p['mil_unit']}\n"
        clean_tooltip += f"Alma Mater: {p['school']}\nMajor: {p['major']}"
        net.add_node(uid, label=" ", title=clean_tooltip, size=n_size, shape=n_shape,
            borderWidth=b_width, color={"background": init_bg, "border": init_border},
            x=float(x), y=float(y))
        neighbors_data = []
        for nb_uid, info in true_neighbors_info.get(uid, {}).items():
            nb_name = next((px['name'] for px in persons if px['uid'] == nb_uid), "Unknown")
            if nb_name != "Unknown":
                neighbors_data.append({"name": nb_name, "type": info["type"], "detail": info["detail"]})
        neighbors_data.sort(key=lambda x: x["name"])
        node_meta[uid] = {
            "name": p["name"], "name_ja": p["name_ja"],
            "school": p["school"], "major": p["major"],
            "gen_unit": p["gen_unit"], "mil_unit": p["mil_unit"],
            "rank": p["rank"],
            "score": t_score, "betweenness": b_score, "bet_rank": b_rank,
            "community": comm_id, "dom_major": dom_major_label,
            "is_war": p["is_war"], "unit731": p["unit731"], "color_group": color_grp, "group": p["group"],
            "purged": p.get("purged", False),
            "field_adj": p.get("field_adj", False),
            "cohort": (f"{(p['birth']//10*10)}s" if p.get("birth") else "?"),
            "color_cohort": cohort_color(p.get("birth")),
            "neighbors_data": neighbors_data, "b_width": b_width,
            "n_size": n_size, "is_broker": is_broker,
            "color_bg": color_bg, "color_comm": color_comm, "color_bet": color_bet
        }
    for u, v, attr in G.edges(data=True):
        etype = attr.get("etype", "unit")
        edge_title = attr.get("title", "")
        if etype == "both": ec, wid = "rgba(192, 132, 252, 0.9)", 3
        elif etype == "unit": ec, wid = "rgba(96, 165, 250, 0.7)", 1.5
        else: ec, wid = "rgba(248, 113, 113, 0.7)", 1.5
        net.add_edge(u, v, color={"color": ec, "highlight": "#ffffff"}, width=wid, smooth=False, hidden=True, title=edge_title, etype=etype)
    options = {
        "physics": {"enabled": False},
        "nodes": {"font": {"size": 0}},
        "edges": {"smooth": False},
        "interaction": {"hover": True, "hoverConnectedEdges": True, "selectConnectedEdges": False}
    }
    net.set_options(json.dumps(options))
    html_path = os.path.join(script_dir, "japan_medical_network.html")
    net.save_graph(html_path)
    nmj = json.dumps(node_meta, ensure_ascii=False)
    school_counts = Counter(p["school"] for p in persons if p["uid"] in connected and p["school"] != "Unknown" and p["is_war"])
    top_schools = school_counts.most_common(20)
    school_legend_html = "<div style='max-height:120px; overflow-y:auto; font-size:12px; margin-bottom:12px; line-height:1.6;'>"
    for sch, cnt in top_schools:
        sc_color = dyn_color(sch)
        school_legend_html += f"<div style='margin-bottom:3px;'><span style='color:{sc_color}; font-size:14px; vertical-align:middle;'>■</span> {sch} <span style='color:#64748b'>({cnt})</span></div>"
    school_legend_html += "</div>"
    comm_counts = Counter(partition.values())
    top_comms = comm_counts.most_common(20)
    comm_legend_html = "<div style='max-height:120px; overflow-y:auto; font-size:12px; margin-bottom:12px; line-height:1.6;'>"
    for c, _ in top_comms:
        if c == -1: continue
        c_col = comm_colors.get(c, "#333")
        c_dom = comm_dom_major.get(c, "Mixed")
        comm_legend_html += f"<div style='margin-bottom:3px;'><span style='color:{c_col}; font-size:14px; vertical-align:middle;'>■</span> Comm {c} <span style='color:#64748b'>({c_dom})</span></div>"
    comm_legend_html += "</div>"
    total_count = len(persons)
    field_count = sum(1 for p in persons if p.get("field_adj"))
    # [edit] Compute camera presets dynamically from actual node coordinates (safe across scale changes)
    def region_center(pred):
        xs = [pos[p["uid"]][0] for p in persons if p["uid"] in pos and pred(p["school_ja"])]
        ys = [pos[p["uid"]][1] for p in persons if p["uid"] in pos and pred(p["school_ja"])]
        if xs: return int(sum(xs)/len(xs)), int(sum(ys)/len(ys))
        return 0, 0
    kanto_x, kanto_y   = region_center(lambda s: bool(re.search(r"東京|慶應|慈恵|日本医|日大|順天堂|昭和|千葉|横浜", s)))
    kansai_x, kansai_y = region_center(lambda s: bool(re.search(r"京都|大阪|阪|神戸|兵庫|岡山", s)))
    colony_x, colony_y = region_center(lambda s: bool(re.search(r"京城|セブランス|ソウル|台北|満州|奉天|大連", s)))
    ui  = f"""
<style>
  body,html{{margin:0;padding:0;overflow:hidden;font-family:'Helvetica Neue',Arial,sans-serif;background:#0f172a;color:#f1f5f9}}
  #mynetwork{{width:100vw!important;height:100vh!important}}
  #cp {{position:absolute; top:20px; left:20px; width:330px; max-height:calc(100vh - 290px); overflow-y:auto; background:rgba(30,41,59,.96); border:1px solid #334155; border-radius:8px; box-shadow:0 10px 25px rgba(0,0,0,.5); padding:18px; z-index:9999}}
  #cp h3 {{margin:0 0 12px; color:#f8fafc; font-size:17px}}
  #si {{width:100%; box-sizing:border-box; padding:9px; border:1px solid #475569; border-radius:6px; background:#0f172a; color:#fff; margin-bottom:8px; font-size:13px}}
  #sr {{max-height:140px; overflow-y:auto; margin-top:6px; margin-bottom:8px;}}
  .si-item {{padding:9px; border-bottom:1px solid #1e293b; cursor:pointer; font-size:12px; color:#cbd5e1}}
  .si-item:hover {{background:#334155; color:#fff}}
  .ctrl-group {{margin-top:12px; padding-top:12px; border-top:1px solid #334155;}}
  .ctrl-label {{font-size:12px; font-weight:bold; color:#94a3b8; display:block; margin-bottom:6px;}}
  .select-css {{width:100%; padding:8px; background:#0f172a; color:#fff; border:1px solid #475569; border-radius:4px; font-size:12px; margin-bottom:10px; cursor:pointer;}}
  .checkbox-container {{display:flex; align-items:center; margin-bottom:8px; font-size:12px; cursor:pointer; color:#e2e8f0;}}
  .checkbox-container input {{margin-right:8px; cursor:pointer;}}
  .btn-primary {{width:100%; padding:9px; background:#3b82f6; color:#fff; border:none; border-radius:6px; cursor:pointer; margin-top:6px; font-weight:bold; transition:0.2s;}}
  .btn-primary:hover {{background:#2563eb;}}
  .btn-outline {{width:100%; padding:8px; background:transparent; color:#3b82f6; border:1px solid #3b82f6; border-radius:6px; cursor:pointer; margin-top:8px; font-size:12px; transition:0.2s;}}
  .btn-outline:hover {{background:rgba(59,130,246,0.1);}}
  .lg-section {{margin-top:14px; padding-top:12px; border-top:1px solid #334155; font-size:11px; line-height:1.8; color:#cbd5e1;}}
  #legend-title {{font-weight:bold; font-size:12px; color:#f8fafc; margin-bottom:6px;}}
  #pp {{position:absolute; top:0; right:0; width:360px; height:100vh; background:rgba(30,41,59,.98); border-left:1px solid #334155; display:flex; flex-direction:column; z-index:9998;}}
  #ph {{padding:24px 20px 16px; background:#1e293b; border-bottom:1px solid #334155; position:relative;}}
  #ph h2 {{margin:0 0 5px; font-size:18px; color:#f8fafc;}}
  #ph p {{margin:0; font-size:12px; color:#94a3b8; line-height:1.6;}}
  #pa {{display:none; flex-direction:column; gap:6px; padding:14px 20px; border-bottom:1px solid #334155}}
  .bt {{padding:8px; font-size:12px; font-weight:bold; border:1px solid #475569; background:#0f172a; color:#cbd5e1; cursor:pointer; border-radius:6px; text-align:left; transition:0.2s}}
  .bt:hover {{background:#334155; color:#fff}}
  #stp {{display:none; padding:14px 20px; background:rgba(15,23,42,0.5); border-bottom:1px solid #334155;}}
  .stat-grid {{display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-top:8px;}}
  .stat-box {{background:#0f172a; border:1px solid #475569; border-radius:6px; padding:10px; text-align:center;}}
  .stat-val {{font-size:20px; font-weight:bold; color:#10b981; margin-bottom:2px;}}
  .stat-val.bet {{color:#f59e0b;}}
  .stat-lbl {{font-size:10px; color:#94a3b8; text-transform:uppercase;}}
  #connected-list-container {{flex-grow:1; overflow-y:auto; padding:10px 20px;}}
  .nb-item {{font-size:12px; color:#cbd5e1; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.05); display:flex; flex-direction:column;}}
  .tag-box {{display:inline-block; padding:2px 6px; border-radius:4px; font-size:9px; font-weight:bold; margin-top:4px; width:max-content;}}
  .tag-unit {{background:rgba(59,130,246,0.2); color:#60a5fa; border:1px solid rgba(59,130,246,0.4);}}
  .tag-major {{background:rgba(239,68,68,0.2); color:#f87171; border:1px solid rgba(239,68,68,0.4);}}
  .tag-both {{background:rgba(168,85,247,0.2); color:#c084fc; border:1px solid rgba(168,85,247,0.4);}}
  #statsModal {{display:none; position:fixed; top:50%; left:50%; transform:translate(-50%, -50%); width:480px; background:rgba(30,41,59,.98); border:1px solid #475569; border-radius:8px; padding:25px; z-index:10001; box-shadow:0 25px 50px -12px rgba(0,0,0,0.7);}}
  #statsModal h3 {{margin-top:0; color:#f8fafc; border-bottom:1px solid #475569; padding-bottom:10px;}}
  .stats-section {{margin-bottom:20px; font-size:13px; color:#e2e8f0; line-height:1.6;}}
  #watermark {{position:absolute; bottom:30px; left:50%; transform:translateX(-50%); background:rgba(15,23,42,0.7); padding:12px 24px; border-radius:30px; border:1px solid #334155; pointer-events:none; z-index:9000; text-align:center;}}
  #watermark h1 {{margin:0; color:#f8fafc; font-size:20px; font-weight:bold; letter-spacing:1px;}}
  #watermark p {{margin:4px 0 0 0; color:#94a3b8; font-size:12px; text-transform:uppercase; letter-spacing:2px;}}
  ::-webkit-scrollbar{{width:4px}}::-webkit-scrollbar-track{{background:#0f172a}}::-webkit-scrollbar-thumb{{background:#475569;border-radius:3px}}
</style>
<div id="watermark">
  <h1>15-Year War Military Medicine SNA</h1>
  <p>Structural Holes & Academic Factions</p>
</div>
<div id="cp">
  <h3>Network Analyzer</h3>
  <div style="background:#1e293b; padding:10px; border-radius:6px; margin-bottom:12px; border:1px solid #475569; text-align:center;">
      <span style="font-size:12px; color:#cbd5e1; display:block; margin-bottom:6px;">
          Total: <b>{total_count}</b> | Mil-Med: <b>{war_count}</b> | Field-adj: <b>{field_count}</b>
      </span>
      <label class="checkbox-container" style="justify-content:center; margin:0; padding:6px; background:#0f172a; border-radius:4px; border:1px solid #3b82f6; color:#60a5fa;">
        <input type="checkbox" id="warOnlyToggle"> ️ Show Military Med Only
      </label>
  </div>
  <input id="si" type="text" placeholder="Search Name, School, Unit, Major" autocomplete="off">
  <div id="sr"></div>
  <button id="rb" class="btn-primary">Reset View</button>
  <div class="ctrl-group">
    <span class="ctrl-label">Node Color Mode</span>
    <select id="colorModeSelect" class="select-css">
      <option value="school">① Alma Mater (School)</option>
      <option value="community">② Louvain Community</option>
      <option value="betweenness">③ Betweenness Centrality</option>
      <option value="group">④ War Involvement (Branch)</option>
      <option value="cohort">⑤ Birth Cohort</option>
    </select>
    <span class="ctrl-label">Network Filters</span>
    <label class="checkbox-container">
      <input type="checkbox" id="brokerToggle"> Highlight Top 20 Brokers (Betweenness)
    </label>
    <label class="checkbox-container">
      <input type="checkbox" id="unit731Toggle"> Highlight Unit 731 / Epidemic
    </label>
    <label class="checkbox-container">
      <input type="checkbox" id="purgedToggle"> ️ Highlight Postwar Purged (追放)
    </label>
    <label class="checkbox-container">
      <input type="checkbox" id="edgeToggle"> ️ Hide 'Major' Edges (Robustness)
    </label>
    <label class="checkbox-container">
      <input type="checkbox" id="fieldAdjToggle"> Include Field-adjacent
    </label>
    <span class="ctrl-label" style="margin-top:10px;">Camera View Presets</span>
    <div style="display:flex; gap:4px; margin-bottom:10px;">
      <button class="btn-outline" style="margin-top:0;" onclick="network.fit({{animation:{{duration:700}} }})">Full</button>
      <button class="btn-outline" style="margin-top:0;" onclick="network.moveTo({{position:{{x:{kanto_x}, y:{kanto_y}}}, scale:0.09, animation:{{duration:700}} }})">Kanto</button>
      <button class="btn-outline" style="margin-top:0;" onclick="network.moveTo({{position:{{x:{kansai_x}, y:{kansai_y}}}, scale:0.09, animation:{{duration:700}} }})">Kansai</button>
      <button class="btn-outline" style="margin-top:0;" onclick="network.moveTo({{position:{{x:{colony_x}, y:{colony_y}}}, scale:0.12, animation:{{duration:700}} }})">Colonies</button>
    </div>
    <span class="ctrl-label" style="margin-top:10px;">️ Figure / Poster</span>
    <div style="display:flex; gap:4px; margin-bottom:6px;">
      <button id="posterToggle" class="btn-outline" style="margin-top:0; border-color:#f59e0b; color:#f59e0b;">️ Poster</button>
      <button id="lightBgToggle" class="btn-outline" style="margin-top:0;">◻️ Light BG</button>
      <button id="posterEdgeToggle" class="btn-outline" style="margin-top:0; background:rgba(59,130,246,0.2);">Edges</button>
    </div>
    <div style="font-size:11px;color:#94a3b8;margin-bottom:2px;">Node size (scale)
      <input id="sizeSlider" type="range" min="4" max="22" value="9" style="width:100%;"></div>
    <div style="font-size:11px;color:#94a3b8;margin-bottom:8px;">Label threshold (smaller → more labels)
      <input id="labelSlider" type="range" min="20" max="100" value="42" style="width:100%;"></div>
    <button id="btnScreenshot" class="btn-outline" style="border-color:#10b981; color:#10b981;">Export PNG (Screenshot)</button>
    <button id="btnStats" class="btn-outline">View Statistical Analysis</button>
  </div>
  <div class="lg-section">
    <div id="legend-title">Alma Mater (Top 20)</div>
    <div id="legend-content"></div>
    <div style="margin-top:12px; padding-top:12px; border-top:1px dashed #475569;">
      <b>Edges</b><br>
      <span style="color:#c084fc; font-weight:bold;">━</span> Purple: <b>Both unit & major</b><br>
      <span style="color:#60a5fa; font-weight:bold;">━</span> Blue: <b>Same military unit</b><br>
      <span style="color:#f87171; font-weight:bold;">━</span> Red: <b>Same major</b><br><br>
      ⭐ <b>Star</b> = Top Brokers (Betweenness)<br>
      <b>Border</b> = Modularity Community
    </div>
  </div>
</div>
<div id="pp">
  <div id="ph">
    <h2 id="pn">Network Map Info</h2>
    <p id="pd">Select a node from the map or use the search bar to view detailed connections and structural metrics.</p>
  </div>
  <div id="pa"></div>
  <div id="stp">
    <div class="stat-grid">
      <div class="stat-box">
        <div class="stat-val" id="stat-deg">0</div>
        <div class="stat-lbl">Degree Centrality</div>
      </div>
      <div class="stat-box">
        <div class="stat-val bet" id="stat-bet">#0</div>
        <div class="stat-lbl">Broker Rank</div>
      </div>
    </div>
  </div>
  <div id="connected-list-container"></div>
</div>
<div id="statsModal">
  <h3>Statistical Validation (computed, not placeholder)</h3>
  <div class="stats-section">
    <b>1. School Homophily (permutation test, n=500)</b><br>
    Military-medicine subnetwork: {len(war_nodes)} nodes · {len(edges_tg)} edges<br>
    Observed same-school edges <b>{obs_h*100:.1f}%</b> vs null mean {null_mean*100:.1f}%
    <div style="width:100%;background:#0f172a;border:1px solid #475569;border-radius:4px;height:14px;position:relative;overflow:hidden;margin-top:5px;">
      <div style="position:absolute;top:0;left:0;height:100%;background:#64748b;width:{null_mean*100:.1f}%;"></div>
      <div style="position:absolute;top:0;left:0;height:100%;background:#10b981;width:{obs_h*100:.1f}%;opacity:0.75;"></div>
    </div>
    <span style="color:#f59e0b;font-weight:bold;">Z = {z_homophily:.2f}</span>
  </div>
  <div class="stats-section">
    <b>2. Modularity Robustness (specialty edges removed)</b><br>
    Full (unit+specialty) <b>{mod_full*100:.1f}%</b> → specialty removed (unit only) <b>{mod_nomajor*100:.1f}%</b>
    <div style="width:100%;background:#0f172a;border:1px solid #475569;border-radius:4px;height:14px;position:relative;overflow:hidden;margin-top:5px;">
      <div style="position:absolute;top:0;left:0;height:100%;background:#3b82f6;width:{mod_full*100:.1f}%;"></div>
      <div style="position:absolute;top:0;left:0;height:100%;background:#ef4444;width:{mod_nomajor*100:.1f}%;opacity:0.85;"></div>
    </div>
    <span style="color:#94a3b8;font-size:11px;">Tests whether shared specialty is the key tie forming the clusters.</span>
  </div>
  <div class="stats-section">
    <b>3. Broker Paradox (top-10 betweenness)</b><br>
    · Overlap with top-10 degree: <b>{brk_overlap}/10</b><br>
    · Multiple specialties: <b>{brk_dual}/10</b><br>
    · From Tokyo Imperial: <b>{brk_todai}/10</b>
  </div>
  <div class="stats-section">
    <b>4. Postwar Accountability (public/teaching purge)</b><br>
    Purged among military-medicine <b>{purged_war} ({purged_war_pct:.1f}%)</b> · among brokers <b>{purged_brk}/10</b><br>
    (total purged {purged_total} · subnetwork density {war_density:.4f})
  </div>
  <button class="btn-primary" onclick="document.getElementById('statsModal').style.display='none'">Close Panel</button>
</div>
<script>
window.addEventListener('load',function(){{
  var meta={nmj};
  var nodesData = network.body.data.nodes;
  var edgesData = network.body.data.edges;
  var colorMode = 'school';
  var highlightBrokers = false;
  var hideMajorEdges = false;
  var warOnlyActive = false;
  var unit731Active = false;
  var purgedActive = false;
  var posterMode = false;
  var lightBg = false;
  var posterEdges = true;
  var fieldAdjActive = false;
  var searchHitIds = null;
  var selectedNodeId = null;
  var leg_school = `{school_legend_html}`;
  var leg_comm = `{comm_legend_html}`;
  var leg_bet = `
    <div style="background:linear-gradient(to right, #0000ff, #ff0000); height:12px; width:100%; border-radius:4px; margin-bottom:5px;"></div>
    <div style="display:flex; justify-content:space-between; font-size:11px; color:#cbd5e1;">
      <span style="color:#60a5fa;">Low</span><span style="color:#f87171;">High (Red)</span>
    </div>`;
  var leg_group = `
      <div style='margin-bottom:3px;'><span style='color:#C0504D; font-size:14px; vertical-align:middle;'>■</span> Army</div>
      <div style='margin-bottom:3px;'><span style='color:#4472C4; font-size:14px; vertical-align:middle;'>■</span> Navy</div>
      <div style='margin-bottom:3px;'><span style='color:#E0A000; font-size:14px; vertical-align:middle;'>■</span> Military Med</div>
      <div style='margin-bottom:3px;'><span style='color:#7B5EA7; font-size:14px; vertical-align:middle;'>■</span> Admin</div>
      <div style='margin-bottom:3px;'><span style='color:#8C8C8C; font-size:14px; vertical-align:middle;'>■</span> Unknown Branch</div>
      <div style='margin-bottom:3px;'><span style='color:#D9923E; font-size:14px; vertical-align:middle;'>■</span> Field-adjacent (indirect)</div>
      <div style='margin-bottom:3px;'><span style='color:#555555; font-size:14px; vertical-align:middle;'>■</span> Civilian</div>`;
  var leg_cohort = `<div style="font-size:11px;color:#cbd5e1;line-height:1.7;">Birth cohort (blue = older → red = younger; brighter = later).</div>`;
  document.getElementById('legend-content').innerHTML = leg_school;
  document.getElementById('btnScreenshot').addEventListener('click', function() {{
      var canvas = document.querySelector('#mynetwork canvas');
      if (canvas) {{
          var dataURL = canvas.toDataURL('image/png');
          var a = document.createElement('a');
          a.href = dataURL; a.download = 'SNA_Export_View.png';
          document.body.appendChild(a); a.click(); document.body.removeChild(a);
      }}
  }});
  function applyVisuals() {{
      var nUpdates = [];
      var connectedToSelected = [];
      if (selectedNodeId) {{
          connectedToSelected = network.getConnectedNodes(selectedNodeId);
          connectedToSelected.push(selectedNodeId);
      }}
      nodesData.getIds().forEach(function(nid) {{
          var m = meta[nid];
          if(!m) return;
          var bg = m.color_bg;
          var commBorder = m.color_comm;
          var nodeWidth = m.b_width || 2;
          if(colorMode === 'community') bg = m.color_comm;
          else if(colorMode === 'betweenness') bg = m.color_bet;
          else if(colorMode === 'group') bg = m.is_war ? m.color_group : (m.field_adj ? "#D9923E" : "#555555");
          else if(colorMode === 'cohort') bg = m.color_cohort;
          var op = 1.0;
          if (warOnlyActive && !m.is_war && !(fieldAdjActive && m.field_adj)) op = 0.05;
          if (purgedActive && !m.purged) op = 0.05;
          if (highlightBrokers && m.bet_rank > 20) op = 0.05;
          if (searchHitIds !== null && !searchHitIds.includes(nid)) op = 0.05;
          if (selectedNodeId !== null && !connectedToSelected.includes(nid)) op = 0.05;
          var finalBg = bg;
          var finalBorder = commBorder;
          if (op < 1.0) {{ finalBg = "rgba(100,116,139,0.05)"; finalBorder = "rgba(71,85,105,0.05)"; }}
          if (unit731Active) {{
              if (m.unit731) {{ finalBorder = "#ff0000"; finalBg = "#ff0000"; nodeWidth = 10; op = 1.0; }}
              else {{ finalBg = "rgba(100,116,139,0.05)"; finalBorder = "rgba(71,85,105,0.05)"; op = 0.05; }}
          }}
          nUpdates.push({{ id: nid, color: {{background: finalBg, border: finalBorder}}, borderWidth: nodeWidth }});
      }});
      nodesData.update(nUpdates);
      // Edge display rule: for readability, show a node's edges only when that node is clicked.
      // (S1 'Show Military Med Only' only dims nodes; edges are revealed by click)
      var eUpdates = [];
      edgesData.get().forEach(function(e) {{
          var isHidden = true;
          if (selectedNodeId) {{
              if (e.from === selectedNodeId || e.to === selectedNodeId) isHidden = false;
          }} else if (posterMode && posterEdges && meta[e.from] && meta[e.to] && meta[e.from].is_war && meta[e.to].is_war) {{
              isHidden = false;  // Poster mode + Edges ON: expose the military-medicine connection structure in the still image
          }}
          if (!isHidden && hideMajorEdges && (e.etype === 'major' || e.etype === 'both')) isHidden = true;
          eUpdates.push({{id: e.id, hidden: isHidden}});
      }});
      edgesData.update(eUpdates);
  }}
  document.getElementById('colorModeSelect').addEventListener('change', function(e) {{
      colorMode = e.target.value;
      var lt = document.getElementById('legend-title');
      var lc = document.getElementById('legend-content');
      if(colorMode === 'school') {{ lt.innerText = "Alma Mater (Top 20)"; lc.innerHTML = leg_school; }}
      else if(colorMode === 'community') {{ lt.innerText = "Communities (Top 20)"; lc.innerHTML = leg_comm; }}
      else if(colorMode === 'betweenness') {{ lt.innerText = "Betweenness Score"; lc.innerHTML = leg_bet; }}
      else if(colorMode === 'group') {{ lt.innerText = "️ War Involvement (Branch)"; lc.innerHTML = leg_group; }}
      else if(colorMode === 'cohort') {{ lt.innerText = "Birth Cohort"; lc.innerHTML = leg_cohort; }}
      applyVisuals();
  }});
  document.getElementById('brokerToggle').addEventListener('change', function(e) {{ highlightBrokers = e.target.checked; applyVisuals(); }});
  document.getElementById('purgedToggle').addEventListener('change', function(e) {{ purgedActive = e.target.checked; applyVisuals(); }});
  document.getElementById('warOnlyToggle').addEventListener('change', function(e) {{
      warOnlyActive = e.target.checked;
      if(warOnlyActive) {{
          var select = document.getElementById('colorModeSelect');
          select.value = 'group';
          select.dispatchEvent(new Event('change'));
      }} else {{ applyVisuals(); }}
  }});
  document.getElementById('unit731Toggle').addEventListener('change', function(e) {{ unit731Active = e.target.checked; applyVisuals(); }});
  // robustness: toggle to hide specialty(major) edges (bind only if the element exists -> avoids crash)
  var _etg = document.getElementById('edgeToggle');
  if (_etg) _etg.addEventListener('change', function(e) {{ hideMajorEdges = e.target.checked; applyVisuals(); }});
  document.getElementById('btnStats').addEventListener('click', function() {{ document.getElementById('statsModal').style.display = 'block'; }});
  // Poster mode: enlarge nodes + label key figures + (for stills) show all military-medicine connections
  var posterScale = 9;   // node-enlargement factor (proportional to sqrt of degree); adjustable via slider
  var labelCut = 42;     // label only nodes at/above this size (larger -> fewer labels)
  function applyPoster() {{
      var ups = [];
      var warIds = [];
      nodesData.getIds().forEach(function(nid) {{
          var m = meta[nid]; if(!m) return;
          if (posterMode) {{
              // [key] Hide civilian nodes -> fit to the military-medicine subnetwork -> fill the screen for visibility
              // But if 'Include Field-adjacent' is on, adjacent candidates appear as small orange dots (secondary ring)
              if (!m.is_war) {{
                  if (fieldAdjActive && m.field_adj) {{
                      warIds.push(nid);
                      ups.push({{id: nid, hidden: false, size: 13, label: " ", font: {{size: 0}} }});
                  }} else {{
                      ups.push({{id: nid, hidden: true}});
                  }}
                  return;
              }}
              warIds.push(nid);
              // Size proportional to sqrt of degree(score) -> more important = proportionally larger
              var sz = 12 + Math.sqrt(m.score) * posterScale;
              if (sz > 200) sz = 200;
              var showLbl = (sz >= labelCut || m.unit731);
              var fsz = showLbl ? Math.max(22, Math.min(sz * 0.65, 70)) : 0;  // font size also proportional to node size
              ups.push({{id: nid, hidden: false, size: sz, label: showLbl ? m.name : " ",
                         font: {{size: fsz, color: lightBg ? '#111827' : '#ffffff', strokeWidth: 6, strokeColor: lightBg ? '#ffffff' : '#000000'}} }});
          }} else {{
              ups.push({{id: nid, hidden: false, size: m.n_size, label: " ", font: {{size: 0}} }});
          }}
      }});
      nodesData.update(ups);
      applyVisuals();
      if (posterMode && warIds.length) network.fit({{nodes: warIds, animation:{{duration:600}}}});
      else network.fit({{animation:{{duration:600}}}});
  }}
  document.getElementById('posterToggle').addEventListener('click', function() {{
      posterMode = !posterMode;
      this.style.background = posterMode ? 'rgba(245,158,11,0.2)' : 'transparent';
      applyPoster();
  }});
  document.getElementById('lightBgToggle').addEventListener('click', function() {{
      lightBg = !lightBg;
      document.getElementById('mynetwork').style.background = lightBg ? '#ffffff' : '#0f172a';
      this.style.background = lightBg ? 'rgba(59,130,246,0.2)' : 'transparent';
      if (posterMode) applyPoster(); else applyVisuals();
  }});
  document.getElementById('sizeSlider').addEventListener('input', function(e) {{
      posterScale = parseFloat(e.target.value);
      if (!posterMode) {{ posterMode = true; document.getElementById('posterToggle').style.background = 'rgba(245,158,11,0.2)'; }}
      applyPoster();
  }});
  document.getElementById('labelSlider').addEventListener('input', function(e) {{
      labelCut = parseFloat(e.target.value);
      if (posterMode) applyPoster();
  }});
  document.getElementById('posterEdgeToggle').addEventListener('click', function() {{
      posterEdges = !posterEdges;
      this.style.background = posterEdges ? 'rgba(59,130,246,0.2)' : 'transparent';
      applyVisuals();
  }});
  document.getElementById('fieldAdjToggle').addEventListener('change', function(e) {{
      fieldAdjActive = e.target.checked;
      if (posterMode) applyPoster(); else applyVisuals();
  }});
  var si = document.getElementById('si'), sr = document.getElementById('sr');
  si.addEventListener('input', function() {{
      var q = si.value.trim().toLowerCase(); sr.innerHTML = '';
      if(!q) {{ searchHitIds = null; applyVisuals(); return; }}
      searchHitIds = [];
      var hits = nodesData.getIds().filter(function(nid) {{
          var m = meta[nid]; if(!m) return false;
          if(warOnlyActive && !m.is_war) return false;
          var isHit = (m.name && m.name.toLowerCase().includes(q)) ||
                 (m.name_ja && m.name_ja.toLowerCase().includes(q)) ||
                 (m.school && m.school.toLowerCase().includes(q)) ||
                 (m.gen_unit && m.gen_unit.toLowerCase().includes(q)) ||
                 (m.mil_unit && m.mil_unit.toLowerCase().includes(q)) ||
                 (m.major && m.major.toLowerCase().includes(q));
          if(isHit) searchHitIds.push(nid);
          return isHit;
      }});
      applyVisuals();
      hits.slice(0, 40).forEach(function(nid) {{
          var m = meta[nid], d = document.createElement('div');
          d.className = 'si-item';
          d.innerHTML = '<b>' + m.name + '</b> (' + m.name_ja + ')<br><span style="font-size:10px">' + m.school + '</span>';
          d.onclick = function() {{ focusNode(nid); }};
          sr.appendChild(d);
      }});
  }});
  function resetRightPanel() {{
      document.getElementById('pn').textContent = "Network Map Info";
      document.getElementById('pd').innerHTML = "Select a node from the map or use the search bar to view detailed connections and structural metrics.";
      document.getElementById('pa').style.display = 'none';
      document.getElementById('stp').style.display = 'none';
      document.getElementById('connected-list-container').innerHTML = '';
  }}
  document.getElementById('rb').addEventListener('click', function() {{
      si.value = ''; sr.innerHTML = '';
      searchHitIds = null; selectedNodeId = null;
      resetRightPanel(); applyVisuals();
      network.fit({{animation:{{duration:700}}}});
  }});
  window.hlGroup = function(field, val) {{
      var matchedNodes = [];
      var vals = val.split(',').map(function(v){{return v.trim();}});
      nodesData.getIds().forEach(function(nid) {{
          var m = meta[nid];
          var hit = false;
          if (m && m[field] && m[field] !== 'Unknown') {{
              var m_vals = m[field].split(',').map(function(v){{return v.trim();}});
              for(var i=0; i<vals.length; i++) {{ if(m_vals.includes(vals[i])) {{ hit = true; break; }} }}
          }}
          if (hit) matchedNodes.push(nid);
      }});
      searchHitIds = matchedNodes; selectedNodeId = null; applyVisuals();
  }};
  function focusNode(uid) {{
      var m = meta[uid]; if(!m) return;
      selectedNodeId = uid; searchHitIds = null;
      document.getElementById('pn').textContent = m.name + " (" + m.name_ja + ")";
      document.getElementById('pd').innerHTML =
          '<b>School:</b> ' + m.school + '<br>' +
          '<b>Community:</b> ID ' + m.community + ' (' + m.dom_major + ')<br>' +
          '<b>Major:</b> ' + m.major + '<br>' +
          (m.is_war ? '️ <b>Branch:</b> ' + m.group + (m.unit731 ? ' · <span style="color:#ff6b6b;">731/防疫</span>' : '') + '<br>' : '') +
          '<b>Gen Affil:</b> ' + m.gen_unit + '<br>' +
          '️ <b>Mil Unit:</b> ' + m.mil_unit + '<br>' +
          '<b>Cohort:</b> ' + m.cohort + (m.purged ? ' · <span style="color:#f59e0b;">️ Postwar purge</span>' : '');
      var pa = document.getElementById('pa');
      pa.style.display = 'flex';
      var safeSchool = m.school ? m.school.replace(/'/g, "\\'") : "";
      var safeMajor = m.major ? m.major.replace(/'/g, "\\'") : "";
      var safeMil = m.mil_unit ? m.mil_unit.replace(/'/g, "\\'") : "";
      pa.innerHTML = '<button class="bt" onclick="hlGroup(\\'school\\',\\'' + safeSchool + '\\')">Highlight Alma Mater Network</button>'
                   + '<button class="bt" onclick="hlGroup(\\'major\\',\\'' + safeMajor + '\\')">Highlight Major Network</button>'
                   + '<button class="bt" onclick="hlGroup(\\'mil_unit\\',\\'' + safeMil + '\\')">Highlight Military Unit Network</button>';
      document.getElementById('stp').style.display = 'block';
      document.getElementById('stat-deg').innerText = m.score;
      document.getElementById('stat-bet').innerText = m.is_war ? '#' + m.bet_rank : 'N/A';
      var listHTML = "<b>Faction Network (" + m.neighbors_data.length + " links)</b><br><br>";
      var displayCount = Math.min(m.neighbors_data.length, 500);
      for(var i=0; i<displayCount; i++) {{
          var nd = m.neighbors_data[i];
          var tagClass = nd.type === 'Both' ? 'tag-both' : (nd.type === 'Unit' ? 'tag-unit' : 'tag-major');
          listHTML += "<div class='nb-item'><span>• " + nd.name + "</span>";
          listHTML += "<span class='tag-box " + tagClass + "' title='" + nd.detail + "'>" + nd.detail + "</span></div>";
      }}
      if(m.neighbors_data.length > 500) listHTML += "<div class='nb-item' style='padding-top:10px;'><i>...and " + (m.neighbors_data.length - 500) + " more</i></div>";
      document.getElementById('connected-list-container').innerHTML = listHTML;
      applyVisuals();
      network.focus(uid, {{scale: 0.9, animation: {{duration:700, easingFunction:'easeInOutQuad'}}}});
  }}
  network.on('click', function(p) {{
      if (p.nodes.length > 0) {{
          var uid = p.nodes[0];
          if(warOnlyActive && meta[uid] && !meta[uid].is_war) return;
          focusNode(uid);
      }} else {{ selectedNodeId = null; resetRightPanel(); applyVisuals(); }}
  }});
  var minimapCanvas = document.createElement('canvas');
  minimapCanvas.id = 'minimap';
  minimapCanvas.width = 330; minimapCanvas.height = 240;
  minimapCanvas.style.cssText = 'position:absolute; bottom:20px; left:20px; z-index:9999; background:rgba(30,41,59,0.96); border:1px solid #334155; border-radius:8px; box-shadow:0 10px 25px rgba(0,0,0,0.5); pointer-events:none;';
  document.body.appendChild(minimapCanvas);
  var mCtx = minimapCanvas.getContext('2d');
  network.on("afterDrawing", function() {{
      var positions = network.getPositions();
      var nodeIds = Object.keys(positions);
      if (nodeIds.length === 0) return;
      var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      nodeIds.forEach(function(id) {{
          var pos = positions[id];
          if(pos.x < minX) minX = pos.x; if(pos.x > maxX) maxX = pos.x;
          if(pos.y < minY) minY = pos.y; if(pos.y > maxY) maxY = pos.y;
      }});
      var w = maxX - minX, h = maxY - minY;
      if (w === 0 || h === 0) return;
      var padX = w * 0.1, padY = h * 0.1;
      minX -= padX; maxX += padX; minY -= padY; maxY += padY;
      w = maxX - minX; h = maxY - minY;
      mCtx.clearRect(0, 0, minimapCanvas.width, minimapCanvas.height);
      mCtx.fillStyle = 'rgba(30,41,59,0.96)';
      mCtx.fillRect(0,0, minimapCanvas.width, minimapCanvas.height);
      var scale = Math.min(minimapCanvas.width / w, minimapCanvas.height / h);
      var offsetX = (minimapCanvas.width - w * scale) / 2;
      var offsetY = (minimapCanvas.height - h * scale) / 2;
      mCtx.fillStyle = "rgba(148, 163, 184, 0.6)";
      nodeIds.forEach(function(id) {{
          var pos = positions[id];
          mCtx.fillRect((pos.x - minX) * scale + offsetX, (pos.y - minY) * scale + offsetY, 2, 2);
      }});
      var viewScale = network.getScale();
      var viewPos = network.getViewPosition();
      var clientW = network.body.container.clientWidth / viewScale;
      var clientH = network.body.container.clientHeight / viewScale;
      var mvx = ((viewPos.x - clientW / 2) - minX) * scale + offsetX;
      var mvy = ((viewPos.y - clientH / 2) - minY) * scale + offsetY;
      mCtx.strokeStyle = "#3b82f6"; mCtx.lineWidth = 2;
      mCtx.strokeRect(mvx, mvy, clientW * scale, clientH * scale);
      mCtx.fillStyle = "rgba(59, 130, 246, 0.15)";
      mCtx.fillRect(mvx, mvy, clientW * scale, clientH * scale);
      mCtx.fillStyle = "#cbd5e1"; mCtx.font = "bold 13px Arial";
      mCtx.fillText("️ Mini Map", 12, 22);
  }});
  setTimeout(function(){{ applyVisuals(); }}, 500);
}});
</script>
"""
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    import re as _re
    html = _re.sub(r'<div id="select-menu"[^>]*>.*?</div>\s*</div>\s*</div>', '', html, flags=_re.DOTALL)
    html = html.replace("</body>", ui + "\n</body>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    gexf_path = os.path.join(script_dir, "japan_medical_network.gexf")
    write_gexf(G, persons, pos, gexf_path)
    print(f"\nDone! HTML generated with all Analytical UI features.")
def write_gexf(G, persons, pos, path):
    uid_map = {p["uid"]: p for p in persons}
    root  = ET.Element("gexf", {"xmlns": "http://gexf.net/1.3", "xmlns:viz": "http://www.gexf.net/1.2draft/viz", "version": "1.3"})
    graph = ET.SubElement(root, "graph", {"mode": "static", "defaultedgetype": "undirected"})
    nodes_el = ET.SubElement(graph, "nodes")
    for uid in G.nodes():
        x, y = pos.get(uid, (0, 0))
        if uid in uid_map:
            p = uid_map[uid]
            n = ET.SubElement(nodes_el, "node", {"id": uid, "label": p["name"]})
            color = hex_to_rgb(dyn_color(p["school"]))
            ET.SubElement(n, "viz:position", {"x": str(float(x)/10), "y": str(float(y)/10), "z": "0.0"})
            ET.SubElement(n, "viz:color", {"r": str(color[0]), "g": str(color[1]), "b": str(color[2]), "a": "1.0"})
            ET.SubElement(n, "viz:size", {"value": "20.0"})
    edges_el = ET.SubElement(graph, "edges")
    for i, (u, v, attr) in enumerate(G.edges(data=True)):
        ET.SubElement(edges_el, "edge", {"id": str(i), "source": u, "target": v})
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
if __name__ == "__main__":
    main()

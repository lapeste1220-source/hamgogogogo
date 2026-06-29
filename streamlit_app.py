# =========================================
#  함창고 수시·정시 검색기 (디자인 리뉴얼 완성본 v3)
#  ✅ 앞 메뉴(등급대 분석 / 추천 탐색기) 그대로
#  ✅ 최저 메뉴는 2027최저모음.csv만 사용 (업로드 UI 없음 / 2025 최저모음 미사용)
#  ✅ 앱 시작 시 최저 파일을 읽지 않음 → 상단 오류 배너 원천 차단
#  ✅ 인코딩/빈줄/헤더 위치 변동/NaN 헤더/엑셀 위장(PK) 방어
#  ✅ 캐시 강제 초기화 URL 지원: ?clear_cache=1
#  🎨 네이비·블루 학술 테마 + 절제된 애니메이션
#  🎨 ① 추천 결과 색상 배지 카드 + "나 vs 합격선" 미니 게이지 바
#  🎨 ② 최저 충족/미충족 ✅/⚠️ 상태 카드 + 판정 근거 표시
#  🎨 ③ 등급대 분석 KPI 요약 카드 + 차트 2단 배치
#  🐛 FIX: HTML 들여쓰기로 인한 코드블록 오인(raw 출력) 문제 → render_html()로 평탄화
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import re
import altair as alt
import io
import html as _html

# =========================================
#   ⚙️ 페이지 설정
# =========================================
st.set_page_config(page_title="함창고 수시·정시 검색기", layout="wide")

# =========================================
#   🎨 전역 디자인 테마
# =========================================
def apply_custom_theme():
    st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

    :root{
        --navy:#16335c; --navy-deep:#0f2444; --accent:#2f6fd0; --accent-soft:#e8f0fb;
        --ink:#1c2733; --muted:#6b7785; --line:#e3e8ef; --card-shadow:0 4px 20px rgba(22,51,92,.07);
        --c-high:#d98324; --c-high-bg:#fdf3e6;
        --c-mid:#2f6fd0;  --c-mid-bg:#e8f0fb;
        --c-safe:#2e8b6f; --c-safe-bg:#e6f5ef;
        --ok:#2e8b6f; --ok-bg:#e6f5ef; --warn:#c0563f; --warn-bg:#fbeae6;
    }
    html, body, [class*="css"]{ font-family:'Pretendard', -apple-system, sans-serif !important; color:var(--ink); }
    .stApp{ background:#f6f8fb; }
    .block-container{ padding-top:2.2rem; animation:fadeInUp .55s ease both; }
    @keyframes fadeInUp{ from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:translateY(0);} }
    @keyframes fadeIn{ from{opacity:0;} to{opacity:1;} }
    h1{ font-weight:800 !important; color:var(--navy) !important; letter-spacing:-.5px; }
    h2, h3{ color:var(--navy-deep) !important; font-weight:700 !important; }
    [data-testid="stHeader"]{ background:transparent; }
    [data-testid="stSidebar"]{ background:linear-gradient(180deg,var(--navy) 0%, var(--navy-deep) 100%); }
    [data-testid="stSidebar"] *{ color:#eaf1fb !important; }
    [data-testid="stSidebar"] .stRadio label{ padding:.45rem .2rem; border-radius:8px; transition:background .2s ease, padding-left .2s ease; }
    [data-testid="stSidebar"] .stRadio label:hover{ background:rgba(255,255,255,.08); padding-left:.6rem; }
    .stButton > button{
        background:var(--accent); color:#fff; border:none; border-radius:10px; padding:.55rem 1.4rem; font-weight:600;
        transition:transform .15s ease, box-shadow .2s ease, background .2s ease; box-shadow:0 2px 8px rgba(47,111,208,.25);
    }
    .stButton > button:hover{ background:var(--navy); transform:translateY(-2px); box-shadow:0 6px 18px rgba(22,51,92,.28); }
    [data-baseweb="input"], .stNumberInput input, .stTextInput input{ border-radius:10px !important; }
    [data-testid="stMetric"]{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:1rem 1.2rem; box-shadow:var(--card-shadow); animation:fadeIn .6s ease both; }
    [data-testid="stMetricValue"]{ color:var(--navy) !important; font-weight:800; }
    [data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:12px; overflow:hidden; box-shadow:var(--card-shadow); }
    .stTabs [data-baseweb="tab-list"]{ gap:6px; border-bottom:2px solid var(--line); }
    .stTabs [data-baseweb="tab"]{ font-weight:600; color:var(--muted); border-radius:10px 10px 0 0; padding:.5rem 1.1rem; transition:color .2s ease, background .2s ease; }
    .stTabs [aria-selected="true"]{ color:var(--accent) !important; background:var(--accent-soft); }

    .hch-hero{
        background:linear-gradient(120deg,var(--navy) 0%, var(--accent) 130%); color:#fff; border-radius:18px;
        padding:1.6rem 2rem; margin-bottom:1.4rem; box-shadow:0 8px 30px rgba(22,51,92,.22); animation:fadeInUp .6s ease both;
    }
    .hch-hero h1{ color:#fff !important; margin:0; font-size:1.9rem; }
    .hch-hero p{ margin:.4rem 0 0; opacity:.9; font-size:.95rem; }
    .hch-section{
        display:flex; align-items:center; gap:.6rem; font-size:1.25rem; font-weight:700; color:var(--navy-deep);
        margin:1.4rem 0 .8rem; padding-left:.7rem; border-left:5px solid var(--accent); animation:fadeInUp .5s ease both;
    }

    /* ① 추천 카드 */
    .rec-grid{ display:grid; grid-template-columns:repeat(auto-fill, minmax(260px,1fr)); gap:14px; margin:.4rem 0 1rem; }
    .rec-card{
        background:#fff; border:1px solid var(--line); border-left:5px solid var(--accent); border-radius:14px;
        padding:1rem 1.1rem; box-shadow:var(--card-shadow); animation:fadeInUp .5s ease both; transition:transform .18s ease, box-shadow .2s ease;
    }
    .rec-card:hover{ transform:translateY(-3px); box-shadow:0 10px 26px rgba(22,51,92,.14); }
    .rec-card.high{ border-left-color:var(--c-high); }
    .rec-card.mid{ border-left-color:var(--c-mid); }
    .rec-card.safe{ border-left-color:var(--c-safe); }
    .rec-badge{ display:inline-block; font-size:.74rem; font-weight:700; letter-spacing:.2px; padding:.18rem .6rem; border-radius:6px; margin-bottom:.55rem; }
    .badge-high{ background:var(--c-high-bg); color:var(--c-high); }
    .badge-mid{ background:var(--c-mid-bg); color:var(--c-mid); }
    .badge-safe{ background:var(--c-safe-bg); color:var(--c-safe); }
    .rec-univ{ font-size:1.08rem; font-weight:800; color:var(--navy); line-height:1.3; }
    .rec-major{ font-size:.9rem; color:var(--ink); margin-top:.15rem; }
    .rec-meta{ font-size:.8rem; color:var(--muted); margin-top:.55rem; }
    .rec-metric{ display:flex; justify-content:space-between; align-items:baseline; margin-top:.7rem; padding-top:.6rem; border-top:1px dashed var(--line); }
    .rec-metric .v{ font-size:1.2rem; font-weight:800; color:var(--navy); }
    .rec-metric .l{ font-size:.76rem; color:var(--muted); }
    .rec-diff{ font-size:.82rem; font-weight:700; }
    .diff-up{ color:var(--c-high); } .diff-mid{ color:var(--c-mid); } .diff-down{ color:var(--c-safe); }
    .gauge{ position:relative; height:13px; margin:.9rem 0 .3rem; }
    .gauge-track{ position:absolute; left:0; right:0; top:4px; height:4px; background:#eef2f7; border-radius:4px; }
    .gauge-fill{ position:absolute; top:4px; height:4px; background:#cdd9ea; border-radius:4px; }
    .gauge-cut{ position:absolute; top:0; width:3px; height:13px; border-radius:2px; transform:translateX(-50%); background:var(--accent); }
    .gauge-cut.high{ background:var(--c-high); } .gauge-cut.mid{ background:var(--c-mid); } .gauge-cut.safe{ background:var(--c-safe); }
    .gauge-me{ position:absolute; top:-1px; width:13px; height:13px; border-radius:50%; transform:translateX(-50%); background:#fff; border:3px solid var(--navy); box-shadow:0 1px 4px rgba(0,0,0,.2); z-index:2; }
    .gauge-cap{ display:flex; justify-content:space-between; font-size:.72rem; color:var(--muted); }
    .gauge-cap .gm{ color:var(--navy); font-weight:700; }

    /* ② 최저 상태 카드 */
    .min-grid{ display:grid; grid-template-columns:repeat(auto-fill, minmax(290px,1fr)); gap:14px; margin:.4rem 0 1rem; }
    .min-card{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; box-shadow:var(--card-shadow); animation:fadeInUp .5s ease both; transition:transform .18s ease, box-shadow .2s ease; }
    .min-card:hover{ transform:translateY(-3px); box-shadow:0 10px 26px rgba(22,51,92,.14); }
    .min-card.ok{ border-top:4px solid var(--ok); } .min-card.warn{ border-top:4px solid var(--warn); }
    .min-status{ font-size:.78rem; font-weight:700; padding:.18rem .6rem; border-radius:6px; display:inline-block; }
    .status-ok{ background:var(--ok-bg); color:var(--ok); } .status-warn{ background:var(--warn-bg); color:var(--warn); }
    .min-univ{ font-size:1.06rem; font-weight:800; color:var(--navy); margin-top:.5rem; }
    .min-major{ font-size:.88rem; color:var(--ink); margin-top:.1rem; }
    .min-tag{ display:inline-block; font-size:.74rem; color:var(--muted); background:#f1f4f8; border-radius:6px; padding:.12rem .5rem; margin:.45rem .3rem 0 0; }
    .min-reason{ font-size:.8rem; margin-top:.55rem; padding:.5rem .65rem; border-radius:8px; line-height:1.45; }
    .min-reason.ok{ background:var(--ok-bg); color:var(--ok); } .min-reason.warn{ background:var(--warn-bg); color:var(--warn); }
    .min-rule{ font-size:.8rem; color:var(--muted); margin-top:.55rem; padding-top:.5rem; border-top:1px dashed var(--line); line-height:1.45; }

    /* ③ KPI 카드 */
    .kpi-grid{ display:grid; grid-template-columns:repeat(4, 1fr); gap:14px; margin:.2rem 0 1.2rem; }
    @media (max-width:880px){ .kpi-grid{ grid-template-columns:repeat(2, 1fr); } }
    .kpi-card{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:1rem 1.15rem; box-shadow:var(--card-shadow); animation:fadeInUp .5s ease both; position:relative; overflow:hidden; }
    .kpi-card::before{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--accent); }
    .kpi-card.k-ok::before{ background:var(--c-safe); } .kpi-card.k-rate::before{ background:var(--c-high); } .kpi-card.k-grade::before{ background:var(--navy); }
    .kpi-label{ font-size:.78rem; color:var(--muted); font-weight:600; }
    .kpi-value{ font-size:1.75rem; font-weight:800; color:var(--navy); line-height:1.15; margin-top:.25rem; }
    .kpi-unit{ font-size:.85rem; font-weight:600; color:var(--muted); margin-left:.15rem; }
    .kpi-sub{ font-size:.74rem; color:var(--muted); margin-top:.3rem; }
    </style>
    """, unsafe_allow_html=True)

apply_custom_theme()

# =========================================
#   🐛 핵심 헬퍼: HTML 평탄화 렌더러
#   들여쓰기(공백 4칸+)가 있으면 Streamlit 마크다운이 코드블록으로 오인 →
#   줄별 공백/줄바꿈을 제거해 한 줄로 만든 뒤 렌더(raw 노출 방지)
# =========================================
def render_html(html: str):
    flat = "".join(line.strip() for line in html.splitlines())
    st.markdown(flat, unsafe_allow_html=True)

def section(title, icon="●"):
    render_html(f'<div class="hch-section">{icon} {title}</div>')

# =========================================
#   Altair 차트 테마 통일
# =========================================
def _hch_altair_theme():
    return {
        "config": {
            "view": {"strokeWidth": 0},
            "font": "Pretendard",
            "axis": {
                "labelColor": "#6b7785", "titleColor": "#16335c",
                "gridColor": "#eef2f7", "domainColor": "#d8dee7",
                "labelFontSize": 12, "titleFontSize": 13,
            },
            "bar": {"cornerRadiusEnd": 4, "color": "#2f6fd0"},
            "range": {"category": ["#16335c", "#2f6fd0", "#5b9bd5", "#f29f3d", "#7a8aa0", "#9ec6f0"]},
        }
    }
alt.themes.register("hch", _hch_altair_theme)
alt.themes.enable("hch")

# =========================================
# ✅ 캐시 강제 초기화 스위치
# =========================================
try:
    qp = st.query_params
    if qp.get("clear_cache") == "1":
        st.cache_data.clear()
        st.cache_resource.clear()
except Exception:
    try:
        qp = st.experimental_get_query_params()
        if qp.get("clear_cache", ["0"])[0] == "1":
            st.cache_data.clear()
            st.cache_resource.clear()
    except Exception:
        pass

# =========================================
#   기본 경로 / 파일명
# =========================================
BASE_DIR = Path(__file__).parent

SUJI_2026_FILE = BASE_DIR / "수시진학관리(2026년2월27일).CSV"
SUJI_2025_FILE = BASE_DIR / "수시진학관리(2025년2월4일).csv"
SUJI_2024_FILE = BASE_DIR / "수시진학관리(2024년2월20일).csv"

SUSI_FILE = BASE_DIR / "2025수시입결.csv"
JEONG_FILE = BASE_DIR / "2025정시입결.csv"

CHOEJEO_FILE = BASE_DIR / "2027최저모음.csv"

LOGO_FILE = BASE_DIR / "hch_logo.png"
PASSWORD = "hamchang123"

# =========================================
#   공통 유틸
# =========================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\n", "").replace(" ", "") for c in df.columns]
    return df

def safe_read_csv(path: Path):
    if not path.exists():
        return None
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return None

def read_suji_2026_csv(path: Path):
    if not path.exists():
        return None
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, header=[0, 1])
            flat_cols = []
            for top, sub in df.columns:
                top = str(top).strip().replace("\n", "").replace(" ", "")
                sub = str(sub).strip().replace("\n", "").replace(" ", "")
                if sub == "" or sub.lower().startswith("unnamed"):
                    flat_cols.append(top)
                else:
                    flat_cols.append(f"{top}_{sub}")
            df.columns = flat_cols
            return df
        except Exception:
            pass
    return None

def get_file_version(path: Path):
    return path.stat().st_mtime if path.exists() else 0.0

def decide_admit(row) -> bool:
    final = str(row.get("최종단계", "")).strip()
    if final.lower() == "nan":
        final = ""
    if "불합격" in final:
        return False
    ok_final = ["최초합격", "충원합격", "추가합격", "추합", "최종합격", "합격"]
    return any(k in final for k in ok_final)

def explain_minimum_rule(rule_text, grades):
    """return: (충족여부 bool, 근거문구 str)"""
    if not rule_text or not isinstance(rule_text, str):
        return False, "최저 기준 정보 없음"

    t = rule_text.replace(" ", "")
    nums = [
        g for g in [
            grades["국어"], grades["수학"], grades["영어"],
            grades["탐1"], grades["탐2"], grades["한국사"]
        ]
        if g > 0
    ]
    if not nums:
        return False, "입력한 등급이 없어 판정 불가 (과목 등급을 입력하세요)"

    m_sum = re.search(r"(?:중)?(\d)개영역?(?:등급)?합(\d+)이내", t)
    if m_sum:
        n = int(m_sum.group(1))
        limit = int(m_sum.group(2))
        nums_sorted = sorted(nums)
        if len(nums_sorted) < n:
            return False, f"{n}개 영역 합 기준이나 입력 영역이 {len(nums_sorted)}개뿐 → 판정 불가"
        s = sum(nums_sorted[:n])
        ok = s <= limit
        pick = "+".join(str(x) for x in nums_sorted[:n])
        return ok, f"상위 {n}개 영역 합 {pick}={s} (기준 {limit} 이내) → {'충족' if ok else '미충족'}"

    m_each = re.search(r"(\d)등급이내", t)
    if m_each:
        limit = int(m_each.group(1))
        worst = max(nums)
        ok = all(g <= limit for g in nums)
        return ok, f"전 영역 {limit}등급 이내 · 내 최저 {worst}등급 → {'충족' if ok else '미충족'}"

    m_each2 = re.search(r"각(\d)등급", t)
    if m_each2:
        limit = int(m_each2.group(1))
        worst = max(nums)
        ok = all(g <= limit for g in nums)
        return ok, f"각 영역 {limit}등급 기준 · 내 최저 {worst}등급 → {'충족' if ok else '미충족'}"

    return False, "자동 해석이 어려운 기준 형식 (원문을 직접 확인하세요)"

def parse_minimum_rule(rule_text, grades):
    ok, _ = explain_minimum_rule(rule_text, grades)
    return ok

def pick_recommendations(df, label_col, diff_col, top_n=3):
    results = []
    high = df[df[label_col] == "상향(도전)"]
    if not high.empty:
        results.append(high.nsmallest(top_n, diff_col))
    mid = df[df[label_col] == "적정"]
    if not mid.empty:
        mid = mid.loc[mid[diff_col].abs().sort_values().index].head(top_n)
        results.append(mid)
    safe = df[df[label_col] == "안전"]
    if not safe.empty:
        results.append(safe.nlargest(top_n, diff_col))
    if not results:
        return pd.DataFrame(columns=df.columns)
    rec = pd.concat(results, ignore_index=True)
    dedup_keys = [c for c in ["대학명", "모집단위"] if c in rec.columns]
    if dedup_keys:
        rec = rec.drop_duplicates(subset=dedup_keys, keep="first")
    return rec

def count_valid_range(series, low, high):
    vals = pd.to_numeric(series, errors="coerce")
    return ((vals >= low) & (vals <= high)).sum()

def pick_best_grade_col(df: pd.DataFrame, candidates, low, high, min_valid=30):
    best_col = None
    best_count = -1
    for c in candidates:
        if c not in df.columns:
            continue
        cnt = count_valid_range(df[c], low, high)
        if cnt > best_count:
            best_count = cnt
            best_col = c
    return best_col if best_count >= min_valid else None

def _esc(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return _html.escape(str(x))

# =========================================
# ✅ ① 추천 결과 카드 렌더러 (+ 미니 게이지)
# =========================================
def render_rec_cards(rec: pd.DataFrame, score_col: str, diff_col: str,
                     detail_col=None, score_label="합격평균내신", lower_is_harder=True):
    label_meta = {
        "상향(도전)": ("high", "badge-high", "상향 · 도전"),
        "적정":       ("mid",  "badge-mid",  "적정"),
        "안전":       ("safe", "badge-safe", "안전"),
    }
    order = {"상향(도전)": 0, "적정": 1, "안전": 2}
    rec = rec.copy()
    rec["_o"] = rec["추천구분"].map(order).fillna(9)
    rec = rec.sort_values(["_o", diff_col])

    if lower_is_harder:
        lo, hi, fmt = 1.0, 9.0, "{:.2f}"
    else:
        lo, hi, fmt = 0.0, 100.0, "{:.1f}"

    def _pos(v):
        try:
            p = (float(v) - lo) / (hi - lo) * 100.0
        except Exception:
            return 50.0
        return max(3.0, min(97.0, p))

    cards = []
    for _, r in rec.iterrows():
        lbl = str(r.get("추천구분", ""))
        cls, badge_cls, badge_txt = label_meta.get(lbl, ("mid", "badge-mid", lbl))
        univ = _esc(r.get("대학명", ""))
        major = _esc(r.get("모집단위", r.get("모집단위명", "")))

        meta_parts = []
        for c in ["전형유형", "전형명", "모집군", detail_col]:
            if c and c in r.index:
                v = _esc(r.get(c, ""))
                if v and v.lower() != "nan":
                    meta_parts.append(v)
        meta = " · ".join(dict.fromkeys(meta_parts))

        try:
            sval = float(r.get(score_col))
            sval_txt = fmt.format(sval)
        except Exception:
            sval, sval_txt = None, _esc(r.get(score_col, "-"))

        try:
            d = float(r.get(diff_col))
        except Exception:
            d = 0.0

        if lower_is_harder:
            if d < -0.05:
                diff_cls, diff_txt = "diff-up", f"도전 {abs(d):.2f}↑"
            elif d > 0.05:
                diff_cls, diff_txt = "diff-down", f"여유 {abs(d):.2f}↓"
            else:
                diff_cls, diff_txt = "diff-mid", "근접"
        else:
            if d > 0.05:
                diff_cls, diff_txt = "diff-up", f"도전 {abs(d):.1f}↑"
            elif d < -0.05:
                diff_cls, diff_txt = "diff-down", f"여유 {abs(d):.1f}↓"
            else:
                diff_cls, diff_txt = "diff-mid", "근접"

        gauge_html = ""
        if sval is not None:
            myv = sval - d
            me_pos = _pos(myv)
            cut_pos = _pos(sval)
            f_lo, f_hi = sorted([me_pos, cut_pos])
            gauge_html = (
                '<div class="gauge">'
                '<div class="gauge-track"></div>'
                f'<div class="gauge-fill" style="left:{f_lo:.1f}%; width:{(f_hi-f_lo):.1f}%;"></div>'
                f'<div class="gauge-cut {cls}" style="left:{cut_pos:.1f}%;"></div>'
                f'<div class="gauge-me" style="left:{me_pos:.1f}%;"></div>'
                '</div>'
                f'<div class="gauge-cap"><span class="gm">나 {fmt.format(myv)}</span>'
                f'<span>합격선 {sval_txt}</span></div>'
            )

        meta_html = f'<div class="rec-meta">{meta}</div>' if meta else ''
        card = (
            f'<div class="rec-card {cls}">'
            f'<span class="rec-badge {badge_cls}">{badge_txt}</span>'
            f'<div class="rec-univ">{univ}</div>'
            f'<div class="rec-major">{major}</div>'
            f'{meta_html}{gauge_html}'
            f'<div class="rec-metric">'
            f'<div><span class="l">{score_label}</span><br><span class="v">{sval_txt}</span></div>'
            f'<div class="rec-diff {diff_cls}">{diff_txt}</div>'
            f'</div></div>'
        )
        cards.append(card)

    render_html(f'<div class="rec-grid">{"".join(cards)}</div>')
    st.caption("● 동그라미 = 내 위치, 세로선 = 합격 평균선 · 막대는 둘 사이 격차")

# =========================================
# ✅ ② 최저 충족/미충족 상태 카드 렌더러 (+ 판정 근거)
# =========================================
def render_choejeo_status(df: pd.DataFrame, status_col: str, reason_col=None,
                          col_jeonhyeong=None, col_method=None, region_col=None):
    total = len(df)
    ok_cnt = int(df[status_col].sum())
    warn_cnt = total - ok_cnt
    rate = (ok_cnt / total * 100) if total else 0.0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("전체 후보", f"{total} 곳")
    with m2:
        st.metric("✅ 충족", f"{ok_cnt} 곳")
    with m3:
        st.metric("⚠️ 미충족", f"{warn_cnt} 곳")
    with m4:
        st.metric("충족 비율", f"{rate:.0f}%")

    d = df.copy().sort_values(status_col, ascending=False)

    cards = []
    for _, r in d.iterrows():
        ok = bool(r.get(status_col, False))
        cls = "ok" if ok else "warn"
        s_cls = "status-ok" if ok else "status-warn"
        s_txt = "✅ 최저 충족" if ok else "⚠️ 미충족"
        univ = _esc(r.get("대학명", ""))
        major = _esc(r.get("모집단위명", r.get("모집단위", "")))

        tags = []
        if region_col and region_col in r.index:
            v = _esc(r.get(region_col, ""))
            if v and v.lower() != "nan":
                tags.append(v)
        for c in [col_jeonhyeong, col_method, "전형세부유형"]:
            if c and c in r.index:
                v = _esc(r.get(c, ""))
                if v and v.lower() != "nan":
                    tags.append(v)
        tags_html = "".join(f'<span class="min-tag">{t}</span>' for t in dict.fromkeys(tags))

        reason_html = ""
        if reason_col and reason_col in r.index:
            reason = _esc(r.get(reason_col, ""))
            if reason:
                reason_html = f'<div class="min-reason {cls}">{reason}</div>'

        rule = _esc(r.get("최저학력기준내용", ""))
        rule_html = f'<div class="min-rule">기준 원문 · {rule}</div>' if rule else ''

        card = (
            f'<div class="min-card {cls}">'
            f'<span class="min-status {s_cls}">{s_txt}</span>'
            f'<div class="min-univ">{univ}</div>'
            f'<div class="min-major">{major}</div>'
            f'<div>{tags_html}</div>'
            f'{reason_html}{rule_html}'
            f'</div>'
        )
        cards.append(card)

    render_html(f'<div class="min-grid">{"".join(cards)}</div>')

# =========================================
# ✅ ③ KPI 요약 카드
# =========================================
def render_kpi_cards(total_apply, total_admit, avg_grade):
    rate = (total_admit / total_apply * 100) if total_apply else 0.0
    avg_txt = f"{avg_grade:.2f}" if avg_grade == avg_grade else "-"
    html = (
        '<div class="kpi-grid">'
        f'<div class="kpi-card"><div class="kpi-label">전체 지원</div>'
        f'<div class="kpi-value">{total_apply:,}<span class="kpi-unit">건</span></div>'
        f'<div class="kpi-sub">검색 조건 기준</div></div>'
        f'<div class="kpi-card k-ok"><div class="kpi-label">합격</div>'
        f'<div class="kpi-value">{total_admit:,}<span class="kpi-unit">건</span></div>'
        f'<div class="kpi-sub">최종 합격 처리 기준</div></div>'
        f'<div class="kpi-card k-rate"><div class="kpi-label">합격률</div>'
        f'<div class="kpi-value">{rate:.1f}<span class="kpi-unit">%</span></div>'
        f'<div class="kpi-sub">합격 / 전체 지원</div></div>'
        f'<div class="kpi-card k-grade"><div class="kpi-label">합격자 평균 내신</div>'
        f'<div class="kpi-value">{avg_txt}<span class="kpi-unit">등급</span></div>'
        f'<div class="kpi-sub">대표등급 평균</div></div>'
        '</div>'
    )
    render_html(html)

# =========================================
# ✅ 최저(2027) 로더
# =========================================
def _choejeo_postprocess_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    def s(x):
        if pd.isna(x):
            return ""
        return str(x).strip()

    header_idx = None
    for i in range(min(200, len(raw))):
        row_vals = [s(v) for v in raw.iloc[i].tolist()]
        if any(v == "대학명" for v in row_vals):
            header_idx = i
            break

    if header_idx is None:
        header = [s(v) for v in raw.iloc[0].tolist()]
        df = raw.iloc[1:].copy()
        df.columns = header
    else:
        header = [s(v) for v in raw.iloc[header_idx].tolist()]
        df = raw.iloc[header_idx + 1:].copy()
        df.columns = header

    keep_cols = [c for c in df.columns if s(c) != ""]
    df = df.loc[:, keep_cols]
    df = normalize_columns(df)
    return df

def _decode_best_effort(data: bytes):
    candidates = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "utf-16", "utf-16-le", "latin1"]
    for enc in candidates:
        try:
            text = data.decode(enc, errors="strict")
            if "대학명" in text:
                return text, enc
        except Exception:
            pass
    for enc in candidates:
        try:
            text = data.decode(enc, errors="ignore")
            if "대학명" in text:
                return text, enc
        except Exception:
            pass
    return data.decode("latin1", errors="ignore"), "latin1"

@st.cache_data(show_spinner=False)
def read_choejeo_2027_cached(file_mtime: float):
    if not CHOEJEO_FILE.exists():
        return None, f"파일이 없습니다: {CHOEJEO_FILE.name}"
    try:
        data = CHOEJEO_FILE.read_bytes()
    except Exception as e:
        return None, f"파일 읽기 실패: {e}"

    if data[:2] == b"PK":
        try:
            df = pd.read_excel(io.BytesIO(data))
            df = normalize_columns(df)
            return df, None
        except Exception as e:
            return None, f"엑셀(PK)로 보이나 읽기 실패: {e}"

    text, used_enc = _decode_best_effort(data)
    if "\x00" in text:
        text = text.replace("\x00", "")

    try:
        raw = pd.read_csv(io.StringIO(text), header=None, engine="python")
        df = _choejeo_postprocess_from_raw(raw)
        if df is None or df.empty:
            return None, f"읽기는 됐지만 데이터가 비어있음(인코딩={used_enc})"
        return df, None
    except Exception as e:
        return None, f"CSV 파싱 실패(인코딩={used_enc}): {e}"

# =========================================
#   데이터 로드
# =========================================
@st.cache_data
def load_data(file_versions):
    suji_list = []
    if SUJI_2026_FILE.exists():
        df26 = read_suji_2026_csv(SUJI_2026_FILE)
        if df26 is not None and not df26.empty:
            df26 = normalize_columns(df26)
            df26["입시연도"] = 2026
            suji_list.append(df26)
    if SUJI_2025_FILE.exists():
        df25 = safe_read_csv(SUJI_2025_FILE)
        if df25 is not None and not df25.empty:
            df25 = normalize_columns(df25)
            df25["입시연도"] = 2025
            suji_list.append(df25)
    if SUJI_2024_FILE.exists():
        df24 = safe_read_csv(SUJI_2024_FILE)
        if df24 is not None and not df24.empty:
            df24 = normalize_columns(df24)
            df24["입시연도"] = 2024
            suji_list.append(df24)

    suji = pd.concat(suji_list, ignore_index=True) if suji_list else None
    susi = safe_read_csv(SUSI_FILE) if SUSI_FILE.exists() else None
    jeong = safe_read_csv(JEONG_FILE) if JEONG_FILE.exists() else None
    susi = normalize_columns(susi) if susi is not None else None
    jeong = normalize_columns(jeong) if jeong is not None else None
    choe = None
    return suji, susi, jeong, choe

file_versions = (
    get_file_version(SUJI_2026_FILE),
    get_file_version(SUJI_2025_FILE),
    get_file_version(SUJI_2024_FILE),
    get_file_version(SUSI_FILE),
    get_file_version(JEONG_FILE),
)

suji_df, susi_df, jeong_df, choe_df = load_data(file_versions)

# =========================================
#   로그인
# =========================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        if LOGO_FILE.exists():
            st.image(str(LOGO_FILE), width=140)
        render_html("""
        <div class="hch-hero" style="text-align:center;">
            <h1>🔒 함창고 수시·정시 검색기</h1>
            <p>보안 접속이 필요합니다 · 인가된 사용자 전용</p>
        </div>
        """)
        pwd = st.text_input("비밀번호", type="password", placeholder="비밀번호를 입력하세요")
        if st.button("접속", use_container_width=True):
            if pwd == PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("비밀번호가 틀렸습니다.")
    st.stop()

# =========================================
#   메인 제목 (히어로)
# =========================================
render_html("""
<div class="hch-hero">
    <h1>함창고 수시·정시 검색기</h1>
    <p>함창고 입결 + 2025 어디가 수시·정시·최저 데이터를 통합 분석 <b>(베타)</b></p>
</div>
""")

# =========================================
#   정시 백분위 컬럼 자동 탐색
# =========================================
JEONG_SCORE_COL = None
if jeong_df is not None:
    cand = [c for c in jeong_df.columns if any(k in c.replace(" ", "") for k in ["백분위", "평균백분위", "반영영역"])]
    JEONG_SCORE_COL = cand[0] if cand else None

# =========================================
#   수시 데이터 전처리
# =========================================
SUJI_HAS_DATA = suji_df is not None and not suji_df.empty
col_9_old = None
col_9_new = None
col_5_new = None

if SUJI_HAS_DATA:
    df_old = suji_df[suji_df["입시연도"] < 2026].copy()
    df_new = suji_df[suji_df["입시연도"] >= 2026].copy()

    old_9_candidates = ["일반등급", "전교과평균등급", "평균등급", "내등급(환산)"]
    for c in old_9_candidates:
        if c in suji_df.columns:
            col_9_old = c
            break

    if col_9_old is None and not df_old.empty:
        old_numeric_candidates = [
            c for c in suji_df.columns
            if ("등급" in str(c)) and ("점수" not in str(c)) and ("5등급" not in str(c))
            and ("한국사" not in str(c)) and ("탐구" not in str(c)) and ("제2외" not in str(c))
            and ("최저" not in str(c)) and ("기준" not in str(c))
        ]
        col_9_old = pick_best_grade_col(df_old, old_numeric_candidates, 1, 9, min_valid=10)

    new_9_candidates = [
        c for c in suji_df.columns
        if ("등급" in str(c)) and ("점수" not in str(c)) and ("5등급" not in str(c))
        and ("한국사" not in str(c)) and ("탐구" not in str(c)) and ("제2외" not in str(c))
        and ("최저" not in str(c)) and ("기준" not in str(c))
    ]
    if not df_new.empty:
        col_9_new = pick_best_grade_col(df_new, new_9_candidates, 1, 9, min_valid=30)

    new_5_candidates = [
        c for c in suji_df.columns
        if ("등급" in str(c)) and ("점수" not in str(c))
        and (("5등급" in str(c)) or ("환산" in str(c)))
        and ("한국사" not in str(c)) and ("탐구" not in str(c)) and ("제2외" not in str(c))
        and ("최저" not in str(c)) and ("기준" not in str(c))
    ]
    if not df_new.empty:
        col_5_new = pick_best_grade_col(df_new, new_5_candidates, 1, 5, min_valid=30)

    suji_df["대표등급_9_old"] = np.nan
    suji_df["대표등급_9_new"] = np.nan
    suji_df["대표등급_5_new"] = np.nan
    if col_9_old:
        suji_df["대표등급_9_old"] = pd.to_numeric(suji_df[col_9_old], errors="coerce")
    if col_9_new:
        suji_df["대표등급_9_new"] = pd.to_numeric(suji_df[col_9_new], errors="coerce")
    if col_5_new:
        suji_df["대표등급_5_new"] = pd.to_numeric(suji_df[col_5_new], errors="coerce")

    suji_df["대표등급"] = np.where(suji_df["입시연도"] >= 2026, suji_df["대표등급_9_new"], suji_df["대표등급_9_old"])
    suji_df["5등급변환내신"] = np.where(suji_df["입시연도"] >= 2026, suji_df["대표등급_5_new"], np.nan)
    suji_df["합격"] = suji_df.apply(decide_admit, axis=1)

# =========================================
#   학생 입력 UI
# =========================================
def get_student_inputs():
    section("내 기본 성적 입력", "1)")
    col1, col2 = st.columns(2)
    with col1:
        my_grade = st.number_input("내신 대표 등급 (전교과·국수영 평균 등급)", min_value=1.0, max_value=9.0, step=0.1, value=3.0)
    with col2:
        mock_percent_input = st.number_input("최근 모의고사 백분위 평균 (없으면 0 입력)", min_value=0.0, max_value=100.0, step=1.0, value=0.0)

    section("희망 대학/학과 입력", "1-1)")
    cu, cm = st.columns(2)
    with cu:
        target_univ = st.text_input("희망 대학 (선택 입력)", "")
    with cm:
        target_major = st.text_input("희망 학과 / 모집단위 (선택 입력)", "")

    section("과목별 등급 입력 (선택, 백분위 자동 추정)", "1-2)")
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        g_kor = st.number_input("국어", 1, 9, 1)
    with r1c2:
        g_math = st.number_input("수학", 1, 9, 1)
    with r1c3:
        g_eng = st.number_input("영어", 1, 9, 1)
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        g_t1 = st.number_input("탐구1", 1, 9, 1)
    with r2c2:
        g_t2 = st.number_input("탐구2", 1, 9, 1)
    with r2c3:
        g_hist = st.number_input("한국사", 1, 9, 1)

    grades = [g for g in [g_kor, g_math, g_eng, g_t1, g_t2] if g > 0]
    mock_percent_est = None
    if grades:
        mapping = {1: 96, 2: 89, 3: 77, 4: 62, 5: 47, 6: 32, 7: 20, 8: 11, 9: 4}
        mock_list = [mapping.get(int(round(g)), 50) for g in grades]
        mock_percent_est = float(np.mean(mock_list))
    mock_percentile = mock_percent_input if mock_percent_input > 0 else mock_percent_est

    region_list = ["서울", "경기", "인천", "부산", "대구", "경북", "충북", "충남"]
    selected_regions = st.multiselect("희망 지역 선택", options=region_list, default=region_list)
    return my_grade, mock_percentile, selected_regions, target_univ, target_major

# =========================================
#   학생부 종합 자가진단
# =========================================
def render_jagajin_inside_tab():
    section("학생부 종합 전형 적합도 자가진단", "🧭")
    st.write("각 문항을 1~5점으로 체크해 주세요.")
    questions = [
        "1) 이수 과목의 다양성과 난도가 충분하다.",
        "2) 교과 성취도가 학년 전체 기준 상위권이다.",
        "3) 자율·진로·동아리 활동을 주도적으로 수행했다.",
        "4) 리더십·배려·공동체·의사소통 역량이 드러난다.",
        "5) 프로젝트·캠페인·보고서 활동 경험이 있다.",
        "6) 독서 활동이 전공과 연계된다.",
        "7) 실패 경험과 극복 과정이 구체적이다.",
        "8) 생활기록부 내용에 대해 명확하게 설명 가능하다.",
        "9) 발표·면접 역량이 우수하다.",
        "10) 나만의 활동 키워드·주제가 일관적이다.",
    ]
    col_left, col_right = st.columns(2)
    scores = []
    with col_left:
        for q in questions[:5]:
            scores.append(st.slider(q, 1, 5, 3))
    with col_right:
        for q in questions[5:]:
            scores.append(st.slider(q, 1, 5, 3))

    total = sum(scores)
    max_score = 5 * len(scores)
    ratio = total / max_score * 100

    section("평가 결과", "●")
    r1, r2 = st.columns(2)
    with r1:
        st.metric("총점", f"{total} / {max_score}")
        st.metric("적합도", f"{ratio:.1f}%")
    with r2:
        if total >= 30:
            level, msg = "적정", "학생부 종합 전형 지원에 적합합니다."
        elif total >= 25:
            level, msg = "보통", "기본 준비는 되어 있으나, 보완이 필요합니다."
        else:
            level, msg = "미흡", "학생부 관리와 전형 전략 재정비가 필요합니다."
        st.subheader(f"종합 평가: {level}")
        st.write(msg)

    df_chart = pd.DataFrame({"문항": [f"Q{i+1}" for i in range(10)], "점수": scores})
    c1, c2 = st.columns(2)
    with c1:
        st.bar_chart(df_chart.iloc[:5].set_index("문항"))
    with c2:
        st.bar_chart(df_chart.iloc[5:].set_index("문항"))

# =========================================
#   뷰 1 : 함창고 등급대 분석
# =========================================
def view_grade_analysis():
    section("함창고 등급대 분석", "📊")

    if not SUJI_HAS_DATA:
        st.error("함창고 수시진학관리 데이터가 없어 분석을 진행할 수 없습니다.")
        return

    df = suji_df.copy().dropna(subset=["대표등급"])
    if df.empty:
        st.info("대표등급 데이터가 없습니다.")
        return

    section("검색 조건", "🔎")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        grade_min, grade_max = st.slider("대표등급 범위", 1.0, 9.0, (1.0, 9.0), 0.1)
    with col2:
        year_opts = sorted(df["입시연도"].dropna().unique().tolist())
        selected_years = st.multiselect("입시 연도", year_opts, default=[year_opts[-1]] if year_opts else [])
    with col3:
        region = st.multiselect("지역 선택", options=sorted(df["지역"].dropna().unique()) if "지역" in df.columns else [])
    with col4:
        univ = st.multiselect("대학 선택", options=sorted(df["대학명"].dropna().unique()) if "대학명" in df.columns else [])

    major_keyword = st.text_input("학과 키워드", "")

    filtered = df[(df["대표등급"] >= grade_min) & (df["대표등급"] <= grade_max)]
    if selected_years:
        filtered = filtered[filtered["입시연도"].isin(selected_years)]
    if region and "지역" in filtered.columns:
        filtered = filtered[filtered["지역"].isin(region)]
    if univ and "대학명" in filtered.columns:
        filtered = filtered[filtered["대학명"].isin(univ)]
    if major_keyword and "모집단위" in filtered.columns:
        filtered = filtered[filtered["모집단위"].astype(str).str.contains(major_keyword, na=False)]

    if filtered.empty:
        st.info("조건에 맞는 데이터가 없습니다.")
        return

    vt_col = "전형유형" if "전형유형" in filtered.columns else ("전형명(대)" if "전형명(대)" in filtered.columns else None)
    if vt_col is None:
        st.error("전형 관련 컬럼을 찾을 수 없습니다.")
        return

    base = filtered.assign(전형분류=lambda d: d[vt_col].astype(str).str.extract("(교과|종합|농어촌)", expand=False).fillna("기타"))

    section("세부유형 필터")
    keyword_input = st.text_input("세부유형 검색 (예: 농어촌 기회)", "")
    if "세부유형" not in base.columns:
        base["세부유형"] = ""
    if keyword_input.strip():
        keys = [k for k in re.split(r"[ ,]+", keyword_input) if k.strip()]
        def match_kw(x):
            x = str(x)
            return all(k in x for k in keys)
        base = base[base["세부유형"].apply(match_kw)]
    if base.empty:
        st.info("세부유형 조건까지 적용한 결과, 데이터가 없습니다.")
        return

    admit_only = base[base["합격"]]

    section("핵심 지표 요약", "📌")
    total_apply = len(base)
    total_admit = int(base["합격"].sum())
    avg_grade = pd.to_numeric(admit_only["대표등급"], errors="coerce").mean() if not admit_only.empty else float("nan")
    render_kpi_cards(total_apply, total_admit, avg_grade)

    base["지원전형"] = base[vt_col].astype(str)
    rate_df = base.groupby("지원전형", as_index=False).agg(전체지원=("합격", "size"), 합격=("합격", "sum"))
    rate_df["합격률_pct"] = (rate_df["합격"] / rate_df["전체지원"] * 100).round(1)
    rate_df = rate_df.sort_values(["합격률_pct", "전체지원"], ascending=False)

    ch_l, ch_r = st.columns(2)
    with ch_l:
        section("전형별 합격률", "📈")
        chart_rate = alt.Chart(rate_df).mark_bar().encode(
            x=alt.X("지원전형:N", sort="-y", title=None),
            y=alt.Y("합격률_pct:Q", title="합격률(%)"),
            color=alt.value("#2f6fd0"),
            tooltip=["지원전형", "전체지원", "합격", alt.Tooltip("합격률_pct:Q", title="합격률(%)")]
        ).properties(height=300)
        st.altair_chart(chart_rate, use_container_width=True)
    with ch_r:
        section("합격자 지역 분포", "🗺️")
        if admit_only.empty or "지역" not in admit_only.columns:
            st.info("합격 데이터 없음")
        else:
            region_count = admit_only.groupby("지역").size().reset_index(name="합격자수").sort_values("합격자수", ascending=False)
            top_region = region_count.iloc[0]["지역"]
            chart = alt.Chart(region_count).mark_bar().encode(
                x=alt.X("지역:O", sort="-y", title=None),
                y=alt.Y("합격자수:Q", title="합격자수"),
                color=alt.condition(alt.datum.지역 == top_region, alt.value("#f29f3d"), alt.value("#2f6fd0")),
                tooltip=["지역", "합격자수"]
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

    with st.expander("전형별 합격률 표로 보기"):
        st.dataframe(rate_df.rename(columns={"합격률_pct": "합격률(%)"}), use_container_width=True, hide_index=True)

    st.markdown("---")
    section("필터 조건에 따른 상세 합격 학과 목록")
    detail = base[base["합격"]].copy()
    if detail.empty:
        st.info("조건에 맞는 합격 학과가 없습니다.")
        return

    detail["이름마스킹"] = detail["이름"].astype(str).str[0] + "OO" if "이름" in detail.columns else "OO"
    detail["지원전형"] = detail[vt_col].astype(str)
    detail["세부유형"] = detail.get("세부유형", "")

    min_cols = [c for c in detail.columns if "최저" in c]
    if min_cols:
        mc = min_cols[0]
        detail["최저"] = detail[mc].fillna("없음").replace("", "없음")
    else:
        detail["최저"] = "없음"

    if "5등급변환내신" in detail.columns:
        detail["5등급변환내신"] = detail["5등급변환내신"].where(detail["입시연도"] >= 2026, "")

    table_cols = ["입시연도", "이름마스킹", "대표등급", "5등급변환내신", "지역", "대학명", "모집단위", "지원전형", "세부유형", "최저"]
    table_cols = [c for c in table_cols if c in detail.columns]
    sort_cols = [c for c in ["입시연도", "대표등급", "대학명", "모집단위"] if c in table_cols]
    st.dataframe(detail[table_cols].sort_values(sort_cols), use_container_width=True, hide_index=True)

# =========================================
#   뷰 2 : 수시·정시 추천 탐색기
# =========================================
def view_recommend():
    section("수시·정시 추천 탐색기", "🎯")
    my_grade, mock_percentile, regions, target_univ, target_major = get_student_inputs()
    tab_su, tab_je, tab_jg = st.tabs(["수시 추천", "정시 추천", "학생부종합 자가진단"])

    with tab_su:
        st.subheader("수시 추천 대학 (함창고 수시 합격 데이터 기반)")
        if not SUJI_HAS_DATA:
            st.warning("수시 합격 데이터가 부족합니다.")
            return

        df = suji_df.copy()
        df = df[df["합격"]].dropna(subset=["대표등급"])
        if "지역" in df.columns and regions:
            df = df[df["지역"].isin(regions)]
        if df.empty:
            st.info("해당 조건에서 추천할 데이터가 없습니다.")
            return

        group_cols = ["대학명", "모집단위"]
        if "전형유형" in df.columns:
            group_cols.append("전형유형")
        if "전형세부유형" in df.columns:
            group_cols.append("전형세부유형")
        elif "세부유형" in df.columns:
            group_cols.append("세부유형")

        agg = df.groupby(group_cols, as_index=False)["대표등급"].mean().rename(columns={"대표등급": "합격평균내신"})
        agg["내신차이(합-입)"] = agg["합격평균내신"] - my_grade

        def label_row(d):
            diff = d["내신차이(합-입)"]
            if diff > 0.3:
                return "안전"
            if diff < -0.3:
                return "상향(도전)"
            return "적정"

        agg["추천구분"] = agg.apply(label_row, axis=1)
        if target_univ:
            agg = agg[agg["대학명"].astype(str).str.contains(target_univ, na=False)]
        if target_major:
            agg = agg[agg["모집단위"].astype(str).str.contains(target_major, na=False)]

        rec = pick_recommendations(agg, "추천구분", "내신차이(합-입)", top_n=3)
        if rec.empty:
            st.info("추천 결과가 없습니다.")
        else:
            detail_col = "전형세부유형" if "전형세부유형" in rec.columns else ("세부유형" if "세부유형" in rec.columns else None)
            render_rec_cards(rec, score_col="합격평균내신", diff_col="내신차이(합-입)",
                             detail_col=detail_col, score_label="합격 평균 내신", lower_is_harder=True)
            with st.expander("표로 보기"):
                cols = ["추천구분", "대학명", "모집단위"]
                if "전형유형" in rec.columns:
                    cols.append("전형유형")
                if detail_col:
                    cols.append(detail_col)
                cols += ["합격평균내신", "내신차이(합-입)"]
                cols = [c for c in cols if c in rec.columns]
                st.dataframe(rec[cols], hide_index=True, use_container_width=True)

    with tab_je:
        st.subheader("정시 추천 대학 (백분위 기반)")
        if jeong_df is None or JEONG_SCORE_COL is None:
            st.warning("정시 입결 데이터 부족")
            return
        if mock_percentile is None:
            st.info("정시 추천을 위해 백분위 입력 또는 등급 입력이 필요합니다.")
            return

        dfj = jeong_df.copy()
        if "지역구분" in dfj.columns and regions:
            dfj = dfj[dfj["지역구분"].isin(regions)]

        dfj[JEONG_SCORE_COL] = pd.to_numeric(dfj[JEONG_SCORE_COL], errors="coerce")
        dfj = dfj.dropna(subset=[JEONG_SCORE_COL])
        if dfj.empty:
            st.warning("해당 지역에서 정시 입결 데이터 없음")
            return

        dfj["정시평균백분위"] = dfj[JEONG_SCORE_COL]
        dfj["백분위차이(합-입)"] = dfj["정시평균백분위"] - mock_percentile

        def label_j(row):
            d = row["백분위차이(합-입)"]
            if d > 3:
                return "상향(도전)"
            if d < -3:
                return "안전"
            return "적정"

        dfj["추천구분"] = dfj.apply(label_j, axis=1)
        if target_univ:
            dfj = dfj[dfj["대학명"].astype(str).str.contains(target_univ, na=False)]
        if target_major:
            dfj = dfj[dfj["모집단위"].astype(str).str.contains(target_major, na=False)]

        recj = pick_recommendations(dfj, "추천구분", "백분위차이(합-입)", top_n=3)
        if recj.empty:
            st.info("정시 추천 결과가 없습니다.")
        else:
            render_rec_cards(recj, score_col="정시평균백분위", diff_col="백분위차이(합-입)",
                             detail_col="전형명", score_label="정시 평균 백분위", lower_is_harder=False)
            with st.expander("표로 보기"):
                colsj = ["추천구분", "대학명", "전형명", "모집군", "모집단위", "정시평균백분위", "백분위차이(합-입)"]
                show_cols = [c for c in colsj if c in recj.columns]
                st.dataframe(recj[show_cols], use_container_width=True, hide_index=True)

    with tab_jg:
        render_jagajin_inside_tab()

# =========================================
#   뷰 3 : 최저 기준으로 대학 찾기
# =========================================
def view_choejeo():
    section("최저 기준으로 대학 찾기", "🧮")

    mtime = get_file_version(CHOEJEO_FILE)
    df, err = read_choejeo_2027_cached(mtime)

    if df is None or df.empty:
        st.error(f"최저 기준 데이터 로딩 실패: {err}")
        return

    region_col = "지역구분" if "지역구분" in df.columns else ("지역" if "지역" in df.columns else None)

    def pick_col(df_, candidates):
        for c in candidates:
            if c in df_.columns:
                return c
        return None

    col_jeonhyeong = pick_col(df, ["전형명", "전형유형", "전형", "전형명(대)", "전형세부유형", "선발유형"])
    col_method = pick_col(df, ["전형방법", "전형방법(상세)", "전형방법_상세", "전형방식", "전형방법내용", "전형방법(요약)", "전형방법_내용"])

    section("내 최저 등급 입력", "1)")
    c1, c2, c3 = st.columns(3)
    with c1:
        g_k = st.number_input("국어", 0, 9, 0)
    with c2:
        g_e = st.number_input("영어", 0, 9, 0)
    with c3:
        g_m = st.number_input("수학", 0, 9, 0)
    d1, d2, d3 = st.columns(3)
    with d1:
        g_t1 = st.number_input("탐구1", 0, 9, 0)
    with d2:
        g_t2 = st.number_input("탐구2", 0, 9, 0)
    with d3:
        g_h = st.number_input("한국사", 0, 9, 0)

    my_grades = {"국어": g_k, "영어": g_e, "수학": g_m, "탐1": g_t1, "탐2": g_t2, "한국사": g_h}

    section("지역 및 키워드 선택", "2)")
    reg = st.multiselect("지역 선택", options=sorted(df[region_col].dropna().unique()) if region_col else [])
    keyword = st.text_input("검색 키워드 (대학명/학과/기준 내용)", "")
    show_only_ok = st.toggle("충족 가능한 대학만 보기", value=True)

    if st.button("검색", type="primary"):
        dff = df.copy()
        if reg and region_col:
            dff = dff[dff[region_col].isin(reg)]
        if keyword:
            key = keyword.replace(" ", "")
            conds = []
            for col in ["대학명", "모집단위명", "최저학력기준내용"]:
                if col in dff.columns:
                    conds.append(dff[col].astype(str).str.contains(key, na=False))
            if conds:
                dff = dff[np.logical_or.reduce(conds)]

        if dff.empty:
            st.info("일치하는 대학이 없습니다.")
            return

        if "최저학력기준내용" not in dff.columns:
            st.error("최저학력기준내용 컬럼을 찾을 수 없습니다.")
            st.write("현재 컬럼:", dff.columns.tolist())
            return

        res = dff["최저학력기준내용"].apply(lambda x: explain_minimum_rule(x, my_grades))
        dff["최저충족가능"] = res.apply(lambda r: r[0])
        dff["판정근거"] = res.apply(lambda r: r[1])

        view_df = dff[dff["최저충족가능"]] if show_only_ok else dff
        if view_df.empty:
            st.info("입력 조건을 충족하는 대학이 없습니다.")
            return

        render_choejeo_status(view_df, status_col="최저충족가능", reason_col="판정근거",
                              col_jeonhyeong=col_jeonhyeong, col_method=col_method, region_col=region_col)

        with st.expander("표로 보기"):
            base_cols = [c for c in ["지역구분", "지역", "대학명", "모집단위명"] if c in view_df.columns]
            extra_cols = []
            if col_jeonhyeong and col_jeonhyeong in view_df.columns:
                extra_cols.append(col_jeonhyeong)
            if col_method and col_method in view_df.columns:
                extra_cols.append(col_method)
            if "전형세부유형" in view_df.columns and "전형세부유형" not in extra_cols:
                extra_cols.append("전형세부유형")
            tail_cols = [c for c in ["최저충족가능", "판정근거", "최저학력기준내용"] if c in view_df.columns]
            show_cols = base_cols + extra_cols + tail_cols
            st.dataframe(view_df[show_cols], hide_index=True, use_container_width=True)

# =========================================
#   사이드바 메뉴
# =========================================
with st.sidebar:
    if LOGO_FILE.exists():
        st.image(str(LOGO_FILE), width=120)
    st.markdown("### 메뉴 선택")
    menu = st.radio("메뉴", ["함창고 등급대 분석", "수시·정시 추천 탐색기", "최저 기준으로 대학 찾기"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div style='font-size:.85rem; opacity:.8;'>제작자 함창고 교사 박호종</div>", unsafe_allow_html=True)

# =========================================
#   라우팅
# =========================================
if menu == "함창고 등급대 분석":
    view_grade_analysis()
elif menu == "수시·정시 추천 탐색기":
    view_recommend()
elif menu == "최저 기준으로 대학 찾기":
    view_choejeo()

st.markdown("---")
st.markdown("<div style='text-align:center; font-size:.85rem; color:gray;'>제작자 함창고 교사 박호종</div>", unsafe_allow_html=True)

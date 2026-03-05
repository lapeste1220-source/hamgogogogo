# =========================================
#        🔒 로그인 + 학교 로고 추가
# =========================================
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import re
import altair as alt

PASSWORD = "hamchang123"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.image("hch_logo.png", width=160)
    st.title("🔒 함창고 수시·정시 검색기 보안 접속")

    pwd = st.text_input("비밀번호를 입력하세요:", type="password")

    if st.button("접속"):
        if pwd == PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("비밀번호가 틀렸습니다.")
    st.stop()


# =========================================
#              기본 설정
# =========================================
st.set_page_config(
    page_title="함창고 수시·정시 검색기",
    layout="wide",
)

st.title("함창고 수시·정시 검색기")
st.caption("함창고 입결 + 2025 어디가 수시·정시·최저 데이터를 통합 분석 (베타)")


DATA_DIR = Path(".")

# CSV 파일 경로
SUJI_2026_FILE = DATA_DIR / "수시진학관리(2026년2월27일).csv"
SUJI_2025_FILE = DATA_DIR / "수시진학관리(2025년2월4일).csv"
SUJI_2024_FILE = DATA_DIR / "수시진학관리(2024년2월20일).csv"
SUSI_FILE = DATA_DIR / "2025수시입결.csv"
JEONG_FILE = DATA_DIR / "2025정시입결.csv"
CHOEJEO_FILE = DATA_DIR / "2025최저모음.csv"

SUSI_GRADE_COL = None
SU_DEPT_AVG = None
JEONG_SCORE_COL = None


# =========================================
#      ★ 대학 그룹 자동 분류
# =========================================
UNIV_GROUP = {
    "서울대학교": "SKY",
    "연세대학교": "SKY",
    "고려대학교": "SKY",

    "부산대학교": "지방거점국립대",
    "경북대학교": "지방거점국립대",
    "전남대학교": "지방거점국립대",
    "충남대학교": "지방거점국립대",
    "강원대학교": "지방거점국립대",

    "성균관대학교": "수도권 주요 사립",
    "한양대학교": "수도권 주요 사립",
    "중앙대학교": "수도권 주요 사립",
    "경희대학교": "수도권 주요 사립",
}

def get_univ_group(name):
    return UNIV_GROUP.get(name, "기타")


# =========================================
#      🔧 최저 기준 판정 함수 (오류 해결)
# =========================================
def parse_minimum_rule(rule_text, grades):
    if not rule_text or not isinstance(rule_text, str):
        return False

    t = rule_text.replace(" ", "")
    nums = [
        g for g in [
            grades["국어"], grades["수학"], grades["영어"],
            grades["탐1"], grades["탐2"], grades["한국사"]
        ]
        if g > 0
    ]
    if not nums:
        return False

    # (1) "2등급이내"
    m_each = re.search(r"(\d)등급이내", t)
    if m_each:
        limit = int(m_each.group(1))
        return all(g <= limit for g in nums)

    # (2) "2개영역합5이내"
    m_sum = re.search(r"(?:중)?(\d)개영역?합(\d+)이내", t)
    if m_sum:
        n = int(m_sum.group(1))
        limit = int(m_sum.group(2))
        nums_sorted = sorted(nums)
        if len(nums_sorted) < n:
            return False
        return sum(nums_sorted[:n]) <= limit

    # (3) "각1등급"
    m_each2 = re.search(r"각(\d)등급", t)
    if m_each2:
        limit = int(m_each2.group(1))
        return all(g <= limit for g in nums)

    return False


# =========================================
#         공통: CSV 컬럼 정규화
# =========================================
def normalize_columns(df):
    df = df.copy()
    df.columns = [c.replace("\n", "").replace(" ", "") for c in df.columns]
    return df


# =========================================
#           데이터 로드
# =========================================
@st.cache_data
def load_data():
    suji_list = []

    if SUJI_2026_FILE.exists():
        df26 = pd.read_csv(SUJI_2026_FILE, encoding="utf-8")
        df26 = normalize_columns(df26)
        df26["입시연도"] = 2026
        suji_list.append(df26)

    if SUJI_2025_FILE.exists():
        df25 = pd.read_csv(SUJI_2025_FILE, encoding="utf-8")
        df25 = normalize_columns(df25)
        df25["입시연도"] = 2025
        suji_list.append(df25)

    if SUJI_2024_FILE.exists():
        df24 = pd.read_csv(SUJI_2024_FILE, encoding="utf-8")
        df24 = normalize_columns(df24)
        df24["입시연도"] = 2024
        suji_list.append(df24)

    suji = pd.concat(suji_list, ignore_index=True) if suji_list else None

    susi = pd.read_csv(SUSI_FILE, encoding="utf-8") if SUSI_FILE.exists() else None
    jeong = pd.read_csv(JEONG_FILE, encoding="utf-8") if JEONG_FILE.exists() else None
    choe = pd.read_csv(CHOEJEO_FILE, encoding="utf-8") if CHOEJEO_FILE.exists() else None

    susi = normalize_columns(susi) if susi is not None else None
    jeong = normalize_columns(jeong) if jeong is not None else None
    choe = normalize_columns(choe) if choe is not None else None

    return suji, susi, jeong, choe


suji_df, susi_df, jeong_df, choe_df = load_data()
# =========================================
#     ✔ 정시 백분위 컬럼 자동 탐색
# =========================================

JEONG_SCORE_COL = None

if jeong_df is not None:
    # 정시 CSV 컬럼명이 다양할 수 있어 유연한 탐색 적용
    cand = [
        c for c in jeong_df.columns
        if any(k in c.replace(" ", "") for k in ["백분위", "평균백분위", "반영영역"])
    ]
    JEONG_SCORE_COL = cand[0] if cand else None

    if JEONG_SCORE_COL is None:
        st.warning("⚠ 정시 CSV에서 백분위 관련 컬럼을 찾을 수 없습니다. 정시 추천이 제한됩니다.")
    else:
        st.write(f"정시 백분위 컬럼 자동 인식됨 → **{JEONG_SCORE_COL}**")

# =========================================
#     ✔ 합격 여부 판정 (함창고 수시진학)
# =========================================
def decide_admit(row):
    reg = str(row.get("등록여부", ""))
    final = str(row.get("최종단계", ""))
    reason = str(row.get("불합격사유", ""))

    bad = ["불합격", "미등록", "탈락", "포기", "최저미충족", "최저미달"]
    if any(b in reason for b in bad):
        return False

    ok_reg = ["등록", "합격"]
    ok_final = ["합격", "최종합격", "추가합격", "추합"]

    if any(o in reg for o in ok_reg):
        return True
    if any(o in final for o in ok_final):
        return True

    return False


# =========================================
#        ✔ 대표등급(전교과 평균 등급)
# =========================================
SUJI_HAS_DATA = suji_df is not None and not suji_df.empty

if SUJI_HAS_DATA:

    grade_cols = [
        c for c in suji_df.columns
        if "등급" in c and not any(x in c for x in ["한국사", "탐구", "제2외"])
    ]

    main_grade_col = None
    for k in ["일반등급", "내등급(환산)", "전교과평균등급", "전교과"]:
        k2 = k.replace(" ", "")
        if k2 in suji_df.columns:
            main_grade_col = k2
            break
    if main_grade_col is None and grade_cols:
        main_grade_col = grade_cols[0]

    if main_grade_col:
        suji_df["대표등급"] = pd.to_numeric(suji_df[main_grade_col], errors="coerce")
    else:
        suji_df["대표등급"] = np.nan

    suji_df["합격"] = suji_df.apply(decide_admit, axis=1)


# =========================================
#          ✔ 학생 입력 UI (추천용)
# =========================================
def get_student_inputs():

    st.markdown("### 1) 내 기본 성적 입력")

    col1, col2 = st.columns(2)
    with col1:
        my_grade = st.number_input(
            "내신 대표 등급 (전교과·국수영 평균 등급)",
            min_value=1.0, max_value=9.0, step=1.0, value=3.0
        )
    with col2:
        mock_percent_input = st.number_input(
            "최근 모의고사 백분위 평균 (없으면 0 입력)",
            min_value=0.0, max_value=100.0, step=1.0, value=0.0
        )

    # -----------------------------------------
    #  희망 대학/모집단위 입력
    # -----------------------------------------
    st.markdown("### 1-1) 희망 대학/학과 입력")
    cu, cm = st.columns(2)
    with cu:
        target_univ = st.text_input("희망 대학 (선택 입력)", "")
    with cm:
        target_major = st.text_input("희망 학과 / 모집단위 (선택 입력)", "")

    # -----------------------------------------
    #  백분위 추정용 개별 등급 입력
    # -----------------------------------------
    # -----------------------------------------
    #  1-2) 과목별 등급 입력 (정렬 수정됨)
    # -----------------------------------------
    st.markdown("### 1-2) 과목별 등급 입력 (선택, 백분위 자동 추정)")

    # 첫 번째 행: 국어, 수학, 영어
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        g_kor = st.number_input("국어", 1, 9, 1)
    with row1_col2:
        g_math = st.number_input("수학", 1, 9, 1)
    with row1_col3:
        g_eng = st.number_input("영어", 1, 9, 1)

    # 두 번째 행: 탐구1, 탐구2, 한국사
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        g_t1 = st.number_input("탐구1", 1, 9, 1)
    with row2_col2:
        g_t2 = st.number_input("탐구2", 1, 9, 1)
    with row2_col3:
        g_hist = st.number_input("한국사", 1, 9, 1)

    # 백분위 추정 계산용 리스트 구성
    grades = [g for g in [g_kor, g_math, g_eng, g_t1, g_t2] if g > 0]

    mock_percent_est = None
    if grades:
        mapping = {1:96, 2:89, 3:77, 4:62, 5:47, 6:32, 7:20, 8:11, 9:4}
        mock_list = [mapping.get(int(round(g)), 50) for g in grades]
        mock_percent_est = np.mean(mock_list)

    mock_percentile = mock_percent_input if mock_percent_input > 0 else mock_percent_est

    # -----------------------------------------
    #  희망 지역 선택
    # -----------------------------------------
    region_list = ["서울", "경기", "인천", "부산", "대구", "경북", "충북", "충남"]
    selected_regions = st.multiselect(
        "희망 지역 선택",
        options=region_list,
        default=region_list
    )

    return my_grade, mock_percentile, selected_regions, target_univ, target_major


# =========================================
#        ✔ 학생부 종합 자가진단
# =========================================
def render_jagajin_inside_tab():

    st.markdown("### 학생부 종합 전형 적합도 자가진단")
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

    # ---------------------------
    #     2단 정렬 UI
    # ---------------------------
    col_left, col_right = st.columns(2)
    scores = []

    with col_left:
        for q in questions[:5]:
            scores.append(st.slider(q, 1, 5, 3))

    with col_right:
        for q in questions[5:]:
            scores.append(st.slider(q, 1, 5, 3))

    # ---------------------------
    #     결과 계산
    # ---------------------------
    total = sum(scores)
    max_score = 5 * len(scores)
    ratio = total / max_score * 100

    st.markdown("### ● 평가 결과")
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

    # ---------------------------
    #     점수 그래프 2단 정렬
    # ---------------------------
    df = pd.DataFrame({
        "문항": [f"Q{i+1}" for i in range(10)],
        "점수": scores
    })

    c1, c2 = st.columns(2)
    with c1:
        st.bar_chart(df.iloc[:5].set_index("문항"))
    with c2:
        st.bar_chart(df.iloc[5:].set_index("문항"))

# =========================================
#      ✔ 뷰 1 : 함창고 등급대 분석
# =========================================
def view_grade_analysis():

    st.header("함창고 등급대 분석")

    if not SUJI_HAS_DATA:
        st.error("함창고 수시진학관리 데이터가 없어 분석을 진행할 수 없습니다.")
        return

    df = suji_df.copy()
    df = df.dropna(subset=["대표등급"])

    # ------------------------------------
    #            필터 UI
    # ------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        min_g = float(np.floor(df["대표등급"].min()))
        max_g = float(np.ceil(df["대표등급"].max()))
        grade_min, grade_max = st.slider(
            "대표등급 범위",
            min_value=min_g, max_value=max_g,
            value=(min_g, max_g), step=1.0
        )

    with col2:
        year_opts = sorted(df["입시연도"].dropna().unique())
        selected_years = st.multiselect("입시 연도", year_opts, default=[year_opts[-1]])

    with col3:
        region = st.multiselect(
            "지역 선택",
            options=sorted(df["지역"].dropna().unique())
        )

    with col4:
        univ = st.multiselect(
            "대학 선택",
            options=sorted(df["대학명"].dropna().unique())
        )

    major_keyword = st.text_input("학과 키워드", "")

    # ------------------------------------
    #           필터 적용
    # ------------------------------------
    filtered = df[(df["대표등급"] >= grade_min) & (df["대표등급"] <= grade_max)]

    if selected_years:
        filtered = filtered[filtered["입시연도"].isin(selected_years)]
    if region:
        filtered = filtered[filtered["지역"].isin(region)]
    if univ:
        filtered = filtered[filtered["대학명"].isin(univ)]
    if major_keyword:
        filtered = filtered[filtered["모집단위"].astype(str).str.contains(major_keyword)]

    if filtered.empty:
        st.info("조건에 맞는 데이터가 없습니다.")
        return

    # 전형 분류 생성
    vt_col = "전형유형" if "전형유형" in filtered.columns else "전형명(대)"
    base = filtered.assign(
        전형분류=lambda d: d[vt_col]
        .astype(str)
        .str.extract("(교과|종합|농어촌)", expand=False)
        .fillna("기타")
    )

    admit_only = base[base["합격"]]

    # ------------------------------------
    #       지역 분포 차트
    # ------------------------------------
    st.subheader("합격자 지역 분포")

    if admit_only.empty:
        st.info("합격 데이터 없음")
    else:
        region_count = (
            admit_only.groupby("지역")
            .size()
            .reset_index(name="합격자수")
            .sort_values("합격자수", ascending=False)
        )

        import altair as alt
        top_region = region_count.iloc[0]["지역"]

        chart = (
            alt.Chart(region_count)
            .mark_bar()
            .encode(
                x=alt.X("지역:O", sort="-y"),
                y="합격자수:Q",
                color=alt.condition(
                    alt.datum.지역 == top_region,
                    alt.value("#ff7f0e"), alt.value("#1f77b4")
                )
            )
        )
        st.altair_chart(chart, use_container_width=True)

    # ------------------------------------
    #       전형 분포 & 최저충족률
    # ------------------------------------
    st.subheader("합격 전형 및 최저 충족률")
    col_l, col_r = st.columns(2)

    # --- 전형 분포 ---
    with col_l:
        st.markdown("#### 전형 분포")
        if not admit_only.empty:
            vt_count = (
                admit_only.groupby("전형분류").size().reset_index(name="합격자수")
            )
            pie = (
                alt.Chart(vt_count)
                .mark_arc()
                .encode(
                    theta="합격자수:Q",
                    color="전형분류:N",
                    tooltip=["전형분류", "합격자수"]
                )
            )
            st.altair_chart(pie, use_container_width=True)
        else:
            st.info("합격 데이터 없음")

    # --- 최저 충족률 ---
    with col_r:
        st.markdown("#### 최저 충족률")
        min_cols = [c for c in base.columns if "최저" in c]
        min_col = min_cols[0] if min_cols else None

        if min_col:
            cond = (
                base[min_col].notna()
                & (base[min_col].astype(str).str.strip() != "")
                & (~base[min_col].astype(str).str.contains("없음"))
            )
            base_min = base[cond]
            if not base_min.empty:
                min_stats = (
                    base_min.groupby("전형분류")["합격"]
                    .mean()
                    .reset_index(name="최저충족률")
                )
                min_stats["최저충족률(%)"] = (min_stats["최저충족률"] * 100).round(1)

                bar = (
                    alt.Chart(min_stats)
                    .mark_bar()
                    .encode(
                        x="전형분류:O",
                        y="최저충족률(%):Q",
                        tooltip=["전형분류", "최저충족률(%)"]
                    )
                )
                st.altair_chart(bar, use_container_width=True)
            else:
                st.info("최저 기준 있는 전형 없음")
        else:
            st.info("최저 기준 컬럼 없음")

    # =========================================
    #           ✔ 상세 표 (핵심)
    # =========================================
    st.markdown("---")
    st.markdown("### 필터 조건에 따른 상세 합격 학과 목록")

    detail = base[base["합격"]].copy()
    if detail.empty:
        st.info("조건에 맞는 합격 학과가 없습니다.")
        return

    # 이름 마스킹
    detail["이름마스킹"] = detail["이름"].astype(str).str[0] + "OO"

    # 지원전형
    if "전형유형" in detail.columns:
        detail["지원전형"] = detail["전형유형"]
    else:
        detail["지원전형"] = detail["전형명(대)"]

    # 세부유형
    detail["세부유형"] = detail.get("세부유형", "")

    # ------------------------------------
    #    ✔ 세부유형 검색 (키워드 AND)
    # ------------------------------------
    keyword_input = st.text_input("세부유형 검색 (예: 농어촌 기회)", "")

    if keyword_input.strip():
        keys = [k for k in re.split(r"[ ,]+", keyword_input) if k.strip()]

        def match_kw(x):
            x = str(x)
            return all(k in x for k in keys)

        detail = detail[detail["세부유형"].apply(match_kw)]

    # 최저 처리
    min_cols = [c for c in detail.columns if "최저" in c]
    if min_cols:
        mc = min_cols[0]
        detail["최저"] = detail[mc].fillna("없음").replace("", "없음")
    else:
        detail["최저"] = "없음"

    table_cols = [
        "입시연도","이름마스킹","대표등급","지역",
        "대학명","모집단위","지원전형","세부유형","최저"
    ]
    table_cols = [c for c in table_cols if c in detail.columns]

    table_df = detail[table_cols].sort_values(
        ["입시연도","대표등급","대학명","모집단위"]
    )

    st.dataframe(table_df, use_container_width=True, hide_index=True)
# =========================================
#       ✔ 추천 공통 유틸 (중복 제거)
# =========================================
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
# =========================================
#    ✔ 뷰 2 : 수시·정시 추천 탐색기
# =========================================
def view_recommend():

    st.header("수시·정시 추천 탐색기")

    # 학생 입력 가져오기
    my_grade, mock_percentile, regions, target_univ, target_major = get_student_inputs()

    tab_su, tab_je, tab_jg = st.tabs(["수시 추천", "정시 추천", "학생부종합 자가진단"])

    # ---------------------------------------------------
    #              ✔ 수시 추천
    # ---------------------------------------------------
    with tab_su:
        st.subheader("수시 추천 대학 (함창고 수시 합격 데이터 기반)")

        if not SUJI_HAS_DATA:
            st.warning("수시 합격 데이터가 부족합니다.")
            return

        df = suji_df.copy()
        df = df[df["합격"]]
        df = df.dropna(subset=["대표등급"])

        if "지역" in df.columns and regions:
            df = df[df["지역"].isin(regions)]

        if df.empty:
            st.info("해당 조건에서 추천할 데이터가 없습니다.")
            return

        # 🔥 세부전형 컬럼 자동 포함
        group_cols = ["대학명", "모집단위", "전형유형"]

        if "전형세부유형" in df.columns:
            group_cols.append("전형세부유형")
        elif "세부유형" in df.columns:
            group_cols.append("세부유형")

        agg = (
            df.groupby(group_cols, as_index=False)["대표등급"]
            .mean()
            .rename(columns={"대표등급": "합격평균내신"})
        )

        agg["내신차이(합-입)"] = agg["합격평균내신"] - my_grade

        # 등급 차이에 따른 추천 레벨
        def label_row(d):
            diff = d["내신차이(합-입)"]
            if diff > 0.3:
                return "안전"
            if diff < -0.3:
                return "상향(도전)"
            return "적정"

        agg["추천구분"] = agg.apply(label_row, axis=1)

        # 희망 대학 / 학과 필터
        if target_univ:
            agg = agg[agg["대학명"].astype(str).str.contains(target_univ)]
        if target_major:
            agg = agg[agg["모집단위"].astype(str).str.contains(target_major)]

        rec = pick_recommendations(agg, "추천구분", "내신차이(합-입)", top_n=3)

        if rec.empty:
            st.info("추천 결과가 없습니다.")
        else:
            # 세부전형 컬럼 자동 추가
            detail_col = (
                "전형세부유형" if "전형세부유형" in rec.columns else
                "세부유형" if "세부유형" in rec.columns else None
            )
            cols = ["추천구분", "대학명", "모집단위", "전형유형"]
            if detail_col:
                cols.append(detail_col)
            cols += ["합격평균내신", "내신차이(합-입)"]

            st.dataframe(rec[cols], hide_index=True, use_container_width=True)


    # ---------------------------------------------------
    #              ✔ 정시 추천
    # ---------------------------------------------------
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
            if d > 3: return "상향(도전)"
            if d < -3: return "안전"
            return "적정"

        dfj["추천구분"] = dfj.apply(label_j, axis=1)

        # 희망 대학/학과 필터
        if target_univ:
            dfj = dfj[dfj["대학명"].astype(str).str.contains(target_univ)]
        if target_major:
            dfj = dfj[dfj["모집단위"].astype(str).str.contains(target_major)]

        recj = pick_recommendations(dfj, "추천구분", "백분위차이(합-입)", top_n=3)

        colsj = ["추천구분", "대학명", "전형명", "모집군", "모집단위",
                 "정시평균백분위", "백분위차이(합-입)"]

        st.dataframe(recj[colsj], use_container_width=True, hide_index=True)

    # ---------------------------------------------------
    #              ✔ 학생부 종합 자가진단
    # ---------------------------------------------------
    with tab_jg:
        render_jagajin_inside_tab()
# =========================================
#    ✔ 뷰 3 : 최저 기준으로 대학 찾기
# =========================================
def view_choejeo():

    st.header("최저 기준으로 대학 찾기")

    if choe_df is None:
        st.error("최저 기준 데이터 파일을 찾을 수 없습니다.")
        return

    st.markdown("### 1) 내 최저 등급 입력")

    c1, c2, c3 = st.columns(3)
    with c1:
        g_k = st.number_input("국어", min_value=0, max_value=9, step=1, value=0)
    with c2:
        g_e = st.number_input("영어", min_value=0, max_value=9, step=1, value=0)
    with c3:
        g_m = st.number_input("수학", min_value=0, max_value=9, step=1, value=0)

    d1, d2, d3 = st.columns(3)
    with d1:
        g_t1 = st.number_input("탐구1", min_value=0, max_value=9, step=1, value=0)
    with d2:
        g_t2 = st.number_input("탐구2", min_value=0, max_value=9, step=1, value=0)
    with d3:
        g_h = st.number_input("한국사", min_value=0, max_value=9, step=1, value=0)

    my_grades = {"국어": g_k, "영어": g_e, "수학": g_m, "탐1": g_t1, "탐2": g_t2, "한국사": g_h}

    st.markdown("### 2) 지역 및 키워드 선택")
    reg = st.multiselect(
        "지역 선택",
        options=sorted(choe_df["지역구분"].dropna().unique())
    )
    keyword = st.text_input("검색 키워드 (대학명/학과/기준 내용)", "")

    if st.button("검색", type="primary"):
        df = choe_df.copy()

        if reg:
            df = df[df["지역구분"].isin(reg)]
        if keyword:
            key = keyword.replace(" ", "")
            df = df[
                df["대학명"].astype(str).str.contains(key)
                | df["모집단위명"].astype(str).str.contains(key)
                | df["최저학력기준내용"].astype(str).str.contains(key)
            ]

        if df.empty:
            st.info("일치하는 대학이 없습니다.")
            return

        df["최저충족가능"] = df["최저학력기준내용"].apply(
            lambda x: parse_minimum_rule(x, my_grades)
        )
        df_ok = df[df["최저충족가능"]]

        if df_ok.empty:
            st.info("입력 조건을 충족하는 대학이 없습니다.")
            return

        cols = ["지역구분", "대학명", "전형세부유형", "모집단위명", "최저학력기준내용"]
        st.dataframe(df_ok[cols], hide_index=True, use_container_width=True)
# =========================================
#              사이드바 메뉴
# =========================================
with st.sidebar:
    st.markdown("### 메뉴 선택")
    menu = st.radio(
        "",
        ["함창고 등급대 분석", "수시·정시 추천 탐색기", "최저 기준으로 대학 찾기"]
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.85rem; color:gray;'>제작자 함창고 교사 박호종</div>",
        unsafe_allow_html=True
    )

# 라우팅
if menu == "함창고 등급대 분석":
    view_grade_analysis()
elif menu == "수시·정시 추천 탐색기":
    view_recommend()
elif menu == "최저 기준으로 대학 찾기":
    view_choejeo()

st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:0.85rem; color:gray;'>제작자 함창고 교사 박호종</div>",
    unsafe_allow_html=True
)










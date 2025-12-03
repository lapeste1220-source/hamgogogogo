import streamlit as st
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import re
import altair as alt

# ---------------- 기본 설정 ----------------
st.set_page_config(
    page_title="함창고 수시·정시 검색기",
    layout="wide",
)

st.title("함창고 수시·정시 검색기")
st.caption("함창고 입결 + 2025 어디가 수시·정시·최저 데이터를 함께 보는 전용 도구 (베타)")

DATA_DIR = Path(".")

SUSI_FILE = DATA_DIR / "2025수시입결.csv"
JEONG_FILE = DATA_DIR / "2025정시입결.csv"
CHOEJEO_FILE = DATA_DIR / "2025최저모음.csv"

# 전역에서 사용할 컬럼 이름들
SUSI_GRADE_COL = None        # 어디가 수시 평균 내신(등급) 컬럼명 (있으면 사용, 없으면 None)
SU_DEPT_AVG = None           # 대학/모집단위별 수시 평균 내신
JEONG_SCORE_COL = None       # 정시 평균 백분위 컬럼명

# ---------------- 공통 유틸 ----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.replace("\n", "").replace(" ", "") for c in df.columns]
    return df


def load_multi_year_suji():
    """
    예: 수시진학관리(2025년2월4일).csv
        수시진학관리(2024년2월20일).csv
    처럼 '수시진학관리(20xx...' 패턴의 파일을 모두 읽어 하나의 DataFrame으로 합친다.
    """
    frames = []
    for path in DATA_DIR.glob("수시진학관리(20*.csv"):
        try:
            m = re.search(r"(20\d{2})", path.name)
            year = int(m.group(1)) if m else None

            df = pd.read_csv(path, encoding="utf-8")
            df = normalize_columns(df)
            df["입시연도"] = year
            frames.append(df)
        except Exception as e:
            print("SUJI 로딩 오류:", path.name, e)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


@st.cache_data
def load_data():
    # 함창고 수시진학 (여러 연도)
    suji = load_multi_year_suji()

    susi = jeong = choe = None
    if SUSI_FILE.exists():
        susi = pd.read_csv(SUSI_FILE, encoding="utf-8")
        susi = normalize_columns(susi)
    if JEONG_FILE.exists():
        jeong = pd.read_csv(JEONG_FILE, encoding="utf-8")
        jeong = normalize_columns(jeong)
    if CHOEJEO_FILE.exists():
        choe = pd.read_csv(CHOEJEO_FILE, encoding="utf-8")
        choe = normalize_columns(choe)
    return suji, susi, jeong, choe


suji_df, susi_df, jeong_df, choe_df = load_data()

# ---------------- 어디가 수시/정시 보조 테이블 ----------------
if susi_df is not None:
    grade_candidates = [
        c for c in susi_df.columns
        if any(k in c for k in ["평균", "평균등급", "내신", "등급"])
    ]
    SUSI_GRADE_COL = grade_candidates[0] if grade_candidates else None

    if SUSI_GRADE_COL is not None and SUSI_GRADE_COL in susi_df.columns:
        su_for_avg = susi_df.copy()
        if "전형세부유형" in su_for_avg.columns:
            mask = su_for_avg["전형세부유형"].astype(str).str.contains("교과")
            su_for_avg = su_for_avg[mask]
        if {"대학명", "모집단위명"}.issubset(su_for_avg.columns):
            SU_DEPT_AVG = (
                su_for_avg
                .groupby(["대학명", "모집단위명"], as_index=False)[SUSI_GRADE_COL]
                .mean()
                .rename(columns={SUSI_GRADE_COL: "수시평균내신"})
            )

if jeong_df is not None:
    cand = [c for c in jeong_df.columns if "반영영역평균백분위" in c.replace(" ", "")]
    JEONG_SCORE_COL = cand[0] if cand else None

# ---------------- 함창고 수시진학 데이터 가공 ----------------
SUJI_HAS_DATA = suji_df is not None and not suji_df.empty

if SUJI_HAS_DATA:
    grade_cols = [c for c in suji_df.columns if "등급" in c and not any(
        x in c for x in ["한국사", "탐구", "제2외"]
    )]
    main_grade_col = None
    for key in ["일반등급", "내등급(환산)", "전교과평균등급", "전교과"]:
        col_norm = key.replace(" ", "")
        if col_norm in suji_df.columns:
            main_grade_col = col_norm
            break
    if main_grade_col is None and grade_cols:
        main_grade_col = grade_cols[0]

    if main_grade_col is not None:
        suji_df["대표등급"] = pd.to_numeric(suji_df[main_grade_col], errors="coerce")
    else:
        suji_df["대표등급"] = np.nan

    def decide_admit(row):
        for col in ["등록여부", "최종단계", "불합격사유"]:
            if col in row.index:
                val = str(row[col])
                if any(k in val for k in ["등록", "합격", "최종합격"]):
                    return True
        return False

    suji_df["합격"] = suji_df.apply(decide_admit, axis=1)


# ---------------- 학생 입력 유틸 ----------------
def get_student_inputs():
    st.markdown("#### 1) 내 기본 성적 입력")
    col1, col2 = st.columns(2)
    with col1:
        my_grade = st.number_input(
            "내신 대표 등급(전교과 또는 국수영 평균)",
            min_value=1.0, max_value=9.0, step=0.1, value=3.0
        )
    with col2:
        mock_input = st.number_input(
            "최근 모의고사 백분위 평균(없으면 0 입력)",
            min_value=0.0, max_value=100.0, step=1.0, value=0.0
        )

    st.markdown("#### 2) 최근 모의고사 등급 입력 (없으면 0으로 두세요)")
    row1 = st.columns(3)
    with row1[0]:
        g_kor = st.number_input("국어 등급", 0.0, 9.0, 0.0, 1.0)
    with row1[1]:
        g_eng = st.number_input("영어 등급", 0.0, 9.0, 0.0, 1.0)
    with row1[2]:
        g_math = st.number_input("수학 등급", 0.0, 9.0, 0.0, 1.0)

    row2 = st.columns(3)
    with row2[0]:
        g_t1 = st.number_input("탐구1 등급", 0.0, 9.0, 0.0, 1.0)
    with row2[1]:
        g_t2 = st.number_input("탐구2 등급", 0.0, 9.0, 0.0, 1.0)
    with row2[2]:
        g_hist = st.number_input("한국사 등급", 0.0, 9.0, 0.0, 1.0)

    if mock_input > 0:
        mock_percentile = mock_input
    else:
        grade_list = [g for g in [g_kor, g_math, g_eng, g_t1, g_t2] if g > 0]
        if grade_list:
            mapping = {1: 96, 2: 89, 3: 77, 4: 62, 5: 47, 6: 32, 7: 20, 8: 11, 9: 4}
            perc = [mapping.get(int(round(g)), 50) for g in grade_list]
            mock_percentile = float(np.mean(perc))
        else:
            mock_percentile = None

    region_options = ["서울", "경기", "인천", "부산", "대구", "경북", "충북", "충남"]
    selected_regions = st.multiselect(
        "희망 지역 선택", options=region_options, default=region_options
    )

    return my_grade, mock_percentile, selected_regions


# ---------------- 학생부종합 자가진단 ----------------
def render_jagajin_inside_tab():
    st.markdown("### 학생부 종합 전형 적합도 자가진단")

    st.write("각 문항에 대해 1점(매우 부족) ~ 5점(매우 우수) 사이에서 선택해 주세요.")

    questions = [
        "1) 이수 과목 수와 난도가 충분히 다양한 편이다.",
        "2) 교과 성취도가 학년 전체에서 상위권에 속한다.",
        "3) 자율/진로/동아리 활동을 지속적·주도적으로 수행했다.",
        "4) 리더십·배려·봉사·의사소통·공동체 역량이 잘 드러난다.",
        "5) 프로젝트·캠페인·보고서 활동 경험이 있다.",
        "6) 독서 활동이 풍부하고, 전공·진로와 연결되어 있다.",
        "7) 실패 경험과 극복 과정이 구체적으로 정리되어 있다.",
        "8) 생활기록부 기입 내용에 대해 자신 있게 설명할 수 있다.",
        "9) 발표·면접·스피치 역량이 뛰어난 편이다.",
        "10) 학교 활동 전체를 관통하는 키워드·주제가 분명하다.",
    ]

    scores = []
    for q in questions:
        scores.append(st.slider(q, 1, 5, 3, key=f"jaga_{q}"))

    total = sum(scores)
    max_score = 5 * len(scores)
    ratio = total / max_score * 100

    st.markdown("#### 결과 요약")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("총점", f"{total} / {max_score}")
        st.metric("적합도(%)", f"{ratio:.1f}%")
    with col2:
        if total >= 30:
            level = "적정"
            msg = "학생부 종합 전형 지원에 비교적 잘 준비된 편입니다."
        elif total >= 25:
            level = "보통"
            msg = "기본적인 준비는 되어 있으나, 몇 가지 보완이 필요합니다."
        else:
            level = "미흡"
            msg = "학생부 관리와 전형 전략을 다시 점검하는 것이 좋겠습니다."
        st.subheader(f"종합 평가: {level}")
        st.write(msg)

    st.markdown("#### 문항별 점수 분포")
    df = pd.DataFrame({"문항": [f"Q{i+1}" for i in range(len(scores))], "점수": scores})

    c1, c2 = st.columns(2)
    half = len(df) // 2
    with c1:
        st.bar_chart(df.iloc[:half].set_index("문항"))
    with c2:
        st.bar_chart(df.iloc[half:].set_index("문항"))


# ---------------- 뷰 1: 함창고 등급대 분석 ----------------
def view_grade_analysis():
    st.header("함창고 등급대 분석")
    if not SUJI_HAS_DATA:
        st.error("함창고 수시진학관리 데이터가 없어 분석을 진행할 수 없습니다.")
        return

    df = suji_df.copy()

    min_g = float(df["대표등급"].min()) if df["대표등급"].notna().any() else 1.0
    max_g = float(df["대표등급"].max()) if df["대표등급"].notna().any() else 6.0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        grade_min, grade_max = st.slider(
            "대표등급 범위 선택",
            min_value=1.0,
            max_value=9.0,
            value=(round(min_g, 1), round(max_g, 1)),
            step=0.5,
        )
    with col2:
        years = sorted(df["입시연도"].dropna().unique()) if "입시연도" in df.columns else []
        selected_years = st.multiselect("입시 연도", options=years, default=years)
    with col3:
        region = st.multiselect("지역 선택", options=sorted(df["지역"].dropna().unique()))
    with col4:
        univ = st.multiselect("대학 선택", options=sorted(df["대학명"].dropna().unique()))
    with col5:
        major = st.text_input("학과(모집단위) 키워드", "")

    filtered = df.copy()
    filtered = filtered[
        (filtered["대표등급"] >= grade_min) & (filtered["대표등급"] <= grade_max)
    ]
    if selected_years:
        filtered = filtered[filtered["입시연도"].isin(selected_years)]
    if region:
        filtered = filtered[filtered["지역"].isin(region)]
    if univ:
        filtered = filtered[filtered["대학명"].isin(univ)]
    if major:
        filtered = filtered[filtered["모집단위"].astype(str).str.contains(major)]

    admit_only = filtered[filtered["합격"]]

    st.subheader("합격자 지역 분포")
    if admit_only.empty:
        st.info("선택한 조건에 해당하는 합격 데이터가 없습니다.")
    else:
        region_count = (
            admit_only.groupby("지역")
            .size()
            .reset_index(name="합격자수")
            .sort_values("합격자수", ascending=False)
        )

        top_region = region_count.iloc[0]["지역"]
        top_count = int(region_count.iloc[0]["합격자수"])

        chart = (
            alt.Chart(region_count)
            .mark_bar()
            .encode(
                x=alt.X("지역:O", sort="-y", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("합격자수:Q"),
                color=alt.condition(
                    alt.datum.지역 == top_region,
                    alt.value("#ff7f0e"),
                    alt.value("#1f77b4"),
                ),
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
        st.markdown(f"**가장 많은 지역: {top_region} (합격 {top_count}명)**")

    st.subheader("합격 전형 분포")
    if admit_only.empty:
        st.info("선택한 조건에 해당하는 합격 데이터가 없습니다.")
    else:
        vt_col = "전형유형" if "전형유형" in admit_only.columns else "전형명(대)"
        base = admit_only.assign(
            전형분류=lambda d: d[vt_col]
            .astype(str)
            .str.extract("(교과|종합|논술|실기)", expand=False)
            .fillna("기타")
        )

        vt_count = (
            base.groupby("전형분류")
            .size()
            .reset_index(name="합격자수")
        )

        pie = (
            alt.Chart(vt_count)
            .mark_arc()
            .encode(
                theta="합격자수:Q",
                color="전형분류:N",
                tooltip=["전형분류", "합격자수"],
            )
        )
        st.altair_chart(pie, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 필터 조건에 따른 상세 합격 학과 목록 (함창고 입결)")

    detail = admit_only.copy()

    def mask_name(x: str) -> str:
        if not isinstance(x, str) or not x:
            return ""
        return x[0] + "OO"

    if "이름" in detail.columns:
        detail["이름마스킹"] = detail["이름"].apply(mask_name)
    else:
        detail["이름마스킹"] = ""

    vt_src = None
    for cand in ["전형유형", "전형명(대)", "전형명"]:
        if cand in detail.columns:
            vt_src = cand
            break

    if "최저학력기준" in detail.columns:
        def make_rule(row):
            vt = str(row[vt_src]) if vt_src else ""
            rule = str(row["최저학력기준"]) if pd.notna(row["최저학력기준"]) else ""
            rule = rule if rule.strip() else "없음"
            return f"{vt} / 최저: {rule}"
        detail["지원전형과최저"] = detail.apply(make_rule, axis=1)
    else:
        detail["지원전형과최저"] = detail[vt_src] if vt_src else ""

    cols_for_table = [
        "입시연도", "이름마스킹", "대표등급", "지역", "대학명", "모집단위", "지원전형과최저"
    ]
    cols_for_table = [c for c in cols_for_table if c in detail.columns]

    if not detail.empty:
        table_df = detail[cols_for_table].sort_values(
            ["입시연도", "대표등급", "대학명", "모집단위"]
            if "입시연도" in cols_for_table else
            ["대표등급", "대학명", "모집단위"]
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)
    else:
        st.info("조건에 맞는 합격 학과가 없습니다.")


# ---------------- 뷰 2: 수시·정시 추천 탐색기 ----------------
def pick_recommendations(df, label_col, diff_col, top_n=2):
    """라벨별로 최대 top_n개씩 뽑되, 없으면 건너뛰기."""
    parts = []
    for label in ["안전", "적정", "상향(도전)"]:
        sub = df[df[label_col] == label].copy()
        if sub.empty:
            continue
        order = sub[diff_col].abs().argsort()
        sub = sub.iloc[order].head(top_n)
        parts.append(sub)
    if parts:
        return pd.concat(parts, ignore_index=True)
    return df


def view_recommend():
    st.header("수시·정시 추천 탐색기")

    my_grade, mock_percentile, regions = get_student_inputs()

    tab_su, tab_je, tab_jg = st.tabs(["수시 추천", "정시 추천", "학생부종합 자가진단"])

    with tab_su:
        st.subheader("수시 추천 대학 (함창고 합격 내신 기준)")

        if not SUJI_HAS_DATA:
            st.warning(
                "아직 우리 학교 수시 합격 내역이 부족하여 추천 계산이 어렵습니다.\n\n"
                "상단 '함창고 등급대 분석' 메뉴에서 전체 합격 데이터를 확인할 수 있습니다."
            )
        else:
            df = suji_df.copy()
            df = df[df["합격"]]

            if "지역" in df.columns and regions:
                df = df[df["지역"].isin(regions)]

            df = df.dropna(subset=["대표등급"])
            if df.empty:
                st.info("선택한 지역에서 합격 내신 데이터가 부족합니다.")
            else:
                group_cols = [c for c in ["대학명", "모집단위", "전형유형"] if c in df.columns]
                agg = (
                    df.groupby(group_cols, as_index=False)["대표등급"]
                    .mean()
                    .rename(columns={"대표등급": "합격평균내신"})
                )

                agg["내신차이(입력-합격)"] = my_grade - agg["합격평균내신"]

                def label_row(d):
                    diff = d["내신차이(입력-합격)"]
                    if diff <= -0.5:
                        return "상향(도전)"
                    if diff >= 0.5:
                        return "안전"
                    return "적정"

                agg["추천구분"] = agg.apply(label_row, axis=1)

                rec = pick_recommendations(agg, "추천구분", "내신차이(입력-합격)", top_n=2)

                cols = (
                    ["추천구분"]
                    + [c for c in ["대학명", "모집단위", "전형유형"] if c in rec.columns]
                    + ["합격평균내신", "내신차이(입력-합격)"]
                )
                if not rec.empty:
                    st.dataframe(rec[cols], use_container_width=True, hide_index=True)
                else:
                    st.info("추천할 만한 데이터를 찾지 못했습니다.")

    with tab_je:
        st.subheader("정시 추천 대학 (모의고사 백분위 기준)")

        if jeong_df is None or JEONG_SCORE_COL is None:
            st.warning("어디가 정시 입결 데이터가 부족하여 정시 추천 계산을 할 수 없습니다.")
        else:
            if mock_percentile is None:
                st.info("정시 추천을 위해서는 최근 모의고사 백분위 평균을 입력하거나, 등급을 몇 과목 이상 입력해 주세요.")
            else:
                dfj = jeong_df.copy()
                if "지역구분" in dfj.columns and regions:
                    dfj = dfj[dfj["지역구분"].isin(regions)]

                dfj[JEONG_SCORE_COL] = pd.to_numeric(dfj[JEONG_SCORE_COL], errors="coerce")
                dfj = dfj.dropna(subset=[JEONG_SCORE_COL])

                if dfj.empty:
                    st.warning("해당 지역에서 정시 입결 데이터가 없습니다.")
                else:
                    dfj["정시평균백분위"] = dfj[JEONG_SCORE_COL]
                    dfj["백분위차이(입력-합격)"] = mock_percentile - dfj["정시평균백분위"]

                    def label_j(row):
                        diff = row["백분위차이(입력-합격)"]
                        if diff <= -5:
                            return "상향(도전)"
                        if diff >= 5:
                            return "안전"
                        return "적정"

                    dfj["추천구분"] = dfj.apply(label_j, axis=1)

                    recj = pick_recommendations(dfj, "추천구분", "백분위차이(입력-합격)", top_n=2)

                    if SU_DEPT_AVG is not None and {"대학명", "모집단위"}.issubset(recj.columns):
                        recj = recj.merge(
                            SU_DEPT_AVG,
                            how="left",
                            left_on=["대학명", "모집단위"],
                            right_on=["대학명", "모집단위명"],
                        )

                    colsj = ["추천구분", "대학명", "전형명", "모집군", "모집단위", "정시평균백분위"]
                    if "수시평균내신" in recj.columns:
                        colsj.append("수시평균내신")

                    st.dataframe(recj[colsj], use_container_width=True, hide_index=True)

    with tab_jg:
        render_jagajin_inside_tab()


# ---------------- 최저 기준으로 대학 찾기 ----------------
def parse_minimum_rule(rule_text, grades):
    if not rule_text or not isinstance(rule_text, str):
        return False

    text = rule_text.replace(" ", "")
    nums = [g for g in [grades["국어"], grades["수학"], grades["영어"], grades["탐1"], grades["탐2"], grades["한국사"]] if g > 0]
    if not nums:
        return False

    m_each = re.search(r"(\d)등급이내", text)
    if m_each:
        limit = int(m_each.group(1))
        return all(g <= limit for g in nums)

    m_sum = re.search(r"(?:중)?(\d)개영역?합(\d+)이내", text)
    if m_sum:
        n = int(m_sum.group(1))
        s_limit = int(m_sum.group(2))
        nums_sorted = sorted(nums)
        if len(nums_sorted) < n:
            return False
        return sum(nums_sorted[:n]) <= s_limit

    m_each2 = re.search(r"각(\d)등급", text)
    if m_each2:
        limit = int(m_each2.group(1))
        return all(g <= limit for g in nums)

    return False


def view_choejeo():
    st.header("최저 기준으로 대학 찾기")

    if choe_df is None:
        st.error("2025 최저 기준 데이터(2025최저모음.csv)를 찾을 수 없습니다.")
        return

    st.markdown("### 1) 내 희망 최저 기준 입력")

    row1 = st.columns(3)
    with row1[0]:
        g_k = st.number_input("국어 최대 등급(0=미사용)", 0.0, 9.0, 0.0, 1.0, key="min_k")
    with row1[1]:
        g_e = st.number_input("영어 최대 등급(0=미사용)", 0.0, 9.0, 0.0, 1.0, key="min_e")
    with row1[2]:
        g_m = st.number_input("수학 최대 등급(0=미사용)", 0.0, 9.0, 0.0, 1.0, key="min_m")

    row2 = st.columns(3)
    with row2[0]:
        g_t1 = st.number_input("탐구1 최대 등급(0=미사용)", 0.0, 9.0, 0.0, 1.0, key="min_t1")
    with row2[1]:
        g_t2 = st.number_input("탐구2 최대 등급(0=미사용)", 0.0, 9.0, 0.0, 1.0, key="min_t2")
    with row2[2]:
        g_h = st.number_input("한국사 최대 등급(0=미사용)", 0.0, 9.0, 0.0, 1.0, key="min_h")

    st.caption("※ 0으로 두면 해당 과목은 최저 기준에서 고려하지 않습니다. 실제 대학별 세부 조건과는 차이가 있을 수 있습니다.")

    colr1, colr2 = st.columns([2, 1])
    with colr1:
        regions = st.multiselect(
            "지역 선택",
            options=sorted(choe_df["지역구분"].dropna().unique()),
            default=None,
        )
    with colr2:
        keyword = st.text_input("검색 키워드 (대학명/모집단위/내용 일부)", "")

    my_grades = {"국어": g_k, "수학": g_m, "영어": g_e, "탐1": g_t1, "탐2": g_t2, "한국사": g_h}

    st.markdown("### 2) 최저 기준에 맞는 대학 검색")

    if st.button("최저 기준에 맞는 대학 검색", type="primary"):
        df = choe_df.copy()
        if regions:
            df = df[df["지역구분"].isin(regions)]
        if keyword:
            pattern = keyword.replace(" ", "")
            df = df[
                df["대학명"].astype(str).str.contains(pattern)
                | df["모집단위명"].astype(str).str.contains(pattern)
                | df["최저학력기준내용"].astype(str).str.contains(pattern)
            ]

        if df.empty:
            st.warning("선택한 조건에 해당하는 최저 기준 데이터가 없습니다.")
            return

        df["최저충족가능"] = df["최저학력기준내용"].apply(
            lambda x: parse_minimum_rule(x, my_grades)
        )
        df_ok = df[df["최저충족가능"]]

        if df_ok.empty:
            st.info("입력한 조건에 맞는 최저 기준 대학을 찾지 못했습니다. (해석 불가 조건은 제외되었습니다.)")
            return

        if SU_DEPT_AVG is not None and {"대학명", "모집단위명"}.issubset(df_ok.columns):
            df_ok = df_ok.merge(SU_DEPT_AVG, how="left", on=["대학명", "모집단위명"])

        cols = [
            "지역구분",
            "대학명",
            "전형세부유형",
            "모집단위명",
            "최저학력기준내용",
        ]
        if "수시평균내신" in df_ok.columns:
            cols.append("수시평균내신")

        cols = [c for c in cols if c in df_ok.columns]
        st.dataframe(df_ok[cols], use_container_width=True, hide_index=True)


# ---------------- 사이드바 메뉴 ----------------
with st.sidebar:
    st.markdown("### 메뉴 선택")
    menu = st.radio(
        "",
        ["함창고 등급대 분석", "수시·정시 추천 탐색기", "최저 기준으로 대학 찾기"],
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.85rem; color:gray;'>제작자 함창고 교사 박호종</div>",
        unsafe_allow_html=True,
    )

# ---------------- 라우팅 ----------------
if menu == "함창고 등급대 분석":
    view_grade_analysis()
elif menu == "수시·정시 추천 탐색기":
    view_recommend()
elif menu == "최저 기준으로 대학 찾기":
    view_choejeo()

st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:0.85rem; color:gray;'>제작자 함창고 교사 박호종</div>",
    unsafe_allow_html=True,
)

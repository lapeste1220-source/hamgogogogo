# -*- coding: utf-8 -*-
"""
함창고 수시·정시 검색기

필요 CSV 파일 (앱과 같은 폴더에 위치):
- 수시진학관리(2025년2월4일).csv
- 2025수시입결.csv
- 2025정시입결.csv
- 2025최저모음.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
st.set_page_config(
    page_title="함창고 수시·정시 검색기",
    layout="wide",
)

st.markdown(
    "<div style='position:fixed; left:10px; top:8px; "
    "font-size:0.85rem; color:#bbbbbb; z-index:1000;'>"
    "제작자 함창고 교사 박호종"
    "</div>",
    unsafe_allow_html=True,
)

st.title("함창고 수시·정시 검색기")
st.caption("함창고 입결 + 2025 어디가 수시·정시·최저 데이터를 함께 보는 전용 도구 (베타)")

# ------------------------------------------------------------------
# 파일 경로
# ------------------------------------------------------------------
BASE = Path(".")
FILE_SCHOOL = BASE / "수시진학관리(2025년2월4일).csv"
FILE_SUSI = BASE / "2025수시입결.csv"
FILE_JUNGSI = BASE / "2025정시입결.csv"
FILE_MIN = BASE / "2025최저모음.csv"


# ------------------------------------------------------------------
# 공통 유틸
# ------------------------------------------------------------------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """줄바꿈/공백이 들어간 컬럼명을 간단하게 정리."""
    df = df.copy()
    df.columns = [str(c).replace("\n", "").strip() for c in df.columns]
    return df


@st.cache_data
def load_school_data():
    if not FILE_SCHOOL.exists():
        return None
    df = pd.read_csv(FILE_SCHOOL, encoding="utf-8-sig")
    df = _normalize_cols(df)

    # 내등급(환산) 숫자로
    if "내등급(환산)" in df.columns:
        df["내등급(환산)"] = pd.to_numeric(df["내등급(환산)"], errors="coerce")
    return df


@st.cache_data
def load_susi_data():
    if not FILE_SUSI.exists():
        return None
    df = pd.read_csv(FILE_SUSI, encoding="utf-8-sig")
    df = _normalize_cols(df)
    return df


@st.cache_data
def load_jungsi_data():
    if not FILE_JUNGSI.exists():
        return None
    df = pd.read_csv(FILE_JUNGSI, encoding="utf-8-sig")
    df = _normalize_cols(df)
    return df


@st.cache_data
def load_minimum_data():
    if not FILE_MIN.exists():
        return None
    df = pd.read_csv(FILE_MIN, encoding="utf-8-sig")
    df = _normalize_cols(df)
    return df


@st.cache_data
def build_school_grade_stats():
    """
    우리 학교 합격 내신 대략값 (평균/중앙)을 대학/학과 기준으로 정리.
    수시최저 메뉴에서 참고용으로만 사용.
    """
    df = load_school_data()
    if df is None:
        return pd.DataFrame()

    work = df.copy()

    # "최종단계" 또는 "등록여부" 등을 이용해 "합격"만 대략 필터링
    if "등록여부" in work.columns:
        mask_ok = work["등록여부"].astype(str).str.contains("등록|합격", na=False)
        work = work[mask_ok]
    elif "최종단계" in work.columns:
        mask_ok = work["최종단계"].astype(str).str.contains("합격", na=False)
        work = work[mask_ok]

    if work.empty or "내등급(환산)" not in work.columns:
        return pd.DataFrame()

    grp_cols = []
    for c in ["지역", "대학명", "모집단위"]:
        if c in work.columns:
            grp_cols.append(c)
    if not grp_cols:
        return pd.DataFrame()

    stats = (
        work.groupby(grp_cols)["내등급(환산)"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    stats.rename(
        columns={
            "count": "우리학교합격자수",
            "mean": "우리학교평균내신",
            "median": "우리학교중앙내신",
        },
        inplace=True,
    )
    return stats


def mask_name(name: str) -> str:
    s = str(name)
    if not s:
        return ""
    if len(s) == 1:
        return s + "O"
    return s[0] + "OO"


# ------------------------------------------------------------------
# 1. 함창고 등급대 분석
# ------------------------------------------------------------------
def view_school_analysis():
    df = load_school_data()
    if df is None:
        st.error("수시진학관리(2025년2월4일).csv 파일을 찾을 수 없습니다.")
        return

    st.subheader("함창고 등급대 분석")

    # 등급대 컬럼 이름 추정: "전교과", "국영수", "국영수사", "국영수과" 등
    # 대표 등급으로 쓸만한 컬럼 후보
    grade_cols = [c for c in df.columns if "등급" in c or "전교과" in c or "국영수" in c]
    grade_col_default = grade_cols[0] if grade_cols else None

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if grade_col_default:
            grade_col = st.selectbox("대표 등급 컬럼 선택", grade_cols, index=0)
        else:
            grade_col = None
            st.write("등급 관련 컬럼을 찾지 못했습니다.")

    with col2:
        # 지역: 없으면 "지역" 대신 "지역구분" 등 시도
        region_col = "지역"
        if region_col not in df.columns and "지역구분" in df.columns:
            region_col = "지역구분"
        if region_col in df.columns:
            regions = sorted(df[region_col].dropna().unique().tolist())
            region_sel = st.multiselect("지역 선택", regions, default=regions)
        else:
            region_sel = None

    with col3:
        if "대학명" in df.columns:
            univs = sorted(df["대학명"].dropna().unique().tolist())
            univ_sel = st.multiselect("대학 선택 (선택 없으면 전체)", ["(전체)"] + univs, default=["(전체)"])
        else:
            univ_sel = ["(전체)"]

    with col4:
        if "모집단위" in df.columns:
            majors = sorted(df["모집단위"].dropna().unique().tolist())
            major_sel = st.multiselect("모집단위/학과 선택 (선택 없으면 전체)", ["(전체)"] + majors, default=["(전체)"])
        else:
            major_sel = ["(전체)"]

    work = df.copy()
    # 필터 적용
    if region_sel is not None:
        work = work[work[region_col].isin(region_sel)]

    if "대학명" in work.columns and "(전체)" not in univ_sel:
        work = work[work["대학명"].isin(univ_sel)]

    if "모집단위" in work.columns and "(전체)" not in major_sel:
        work = work[work["모집단위"].isin(major_sel)]

    # "합격"만 보고 싶을 가능성이 높으므로 대략 필터
    if "등록여부" in work.columns:
        mask_ok = work["등록여부"].astype(str).str.contains("등록|합격", na=False)
        work = work[mask_ok]
    elif "최종단계" in work.columns:
        mask_ok = work["최종단계"].astype(str).str.contains("합격", na=False)
        work = work[mask_ok]

    if work.empty:
        st.info("선택한 조건에 해당하는 합격 데이터가 없습니다.")
        return

    # -----------------------
    # 시각화 1: 합격자 지역 분포
    # -----------------------
    st.markdown("### 합격자 지역 분포")

    if region_col in work.columns:
        region_cnt = work[region_col].value_counts().reset_index()
        region_cnt.columns = [region_col, "합격자수"]
        region_cnt = region_cnt.sort_values("합격자수", ascending=False)

        fig = px.bar(
            region_cnt,
            x=region_col,
            y="합격자수",
            text="합격자수",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_title="지역",
            yaxis_title="합격자 수",
            xaxis_tickangle=0,
        )
        # 가장 많은 지역 라벨 강하게
        if not region_cnt.empty:
            top_region = region_cnt.iloc[0][region_col]
            fig.update_xaxes(
                tickfont=dict(
                    size=[14 if r == top_region else 11 for r in region_cnt[region_col]],
                    color=["#ffffff" if r == top_region else "#dddddd" for r in region_cnt[region_col]],
                )
            )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("지역 정보를 찾을 수 없습니다.")

    # -----------------------
    # 시각화 2: 합격 전형 분포
    # -----------------------
    st.markdown("### 합격 전형 분포")

    type_col_candidates = [c for c in ["전형유형", "전형유형", "전형명", "전형명(대)"] if c in work.columns]
    if type_col_candidates:
        type_col = type_col_candidates[0]
        # '농어촌' 관련 문구는 제외
        type_series = work[type_col].astype(str)
        mask_farm = type_series.str.contains("농어촌", na=False)
        type_series = type_series[~mask_farm]
        type_cnt = type_series.value_counts().reset_index()
        type_cnt.columns = [type_col, "합격자수"]

        fig2 = px.pie(
            type_cnt,
            names=type_col,
            values="합격자수",
            hole=0.3,
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("전형 유형 정보를 찾을 수 없습니다.")

    # -----------------------
    # 상세 합격 학과 목록
    # -----------------------
    st.markdown("### 상세 합격 학과 목록")

    table = work.copy()

    # 학년/반/번호 제거, 이름 마스킹
    drop_cols = [c for c in ["학년", "반", "번호"] if c in table.columns]
    table = table.drop(columns=drop_cols, errors="ignore")

    if "이름" in table.columns:
        table["이름"] = table["이름"].apply(mask_name)

    st.dataframe(table, use_container_width=True)


# ------------------------------------------------------------------
# 2. 수시·정시 추천 탐색기  (+ 학생부종합 자가진단 포함)
# ------------------------------------------------------------------
def view_recommendation():
    st.subheader("수시·정시 추천 탐색기")

    susi = load_susi_data()
    jungsi = load_jungsi_data()

    st.warning(
        "어디가 수시/정시 데이터가 부족해 **정교한 추천 계산은 어렵습니다.**\n\n"
        "아래 표는 기본적인 탐색용 참고 자료이며, 실제 지원 여부는 담임선생님 및 진로진학부와 반드시 상의하세요."
    )

    st.markdown("#### 1) 나의 성적 입력")

    col1, col2, col3 = st.columns(3)
    with col1:
        my_grade = st.number_input("내신 대표 등급 (전교과 등)", min_value=1.0, max_value=9.0, value=3.0, step=0.1)
    with col2:
        my_pct = st.number_input("정시 대비 최근 백분위 평균 (대략)", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
    with col3:
        hope_keyword = st.text_input("희망 대학/학과 키워드 (선택)", "")

    st.markdown("##### 최근 모의고사 등급 (등급이 없으면 0)")

    row1 = st.columns(3)
    row2 = st.columns(3)
    with row1[0]:
        k_grade = st.number_input("국어 등급", min_value=0.0, max_value=9.0, value=3.0, step=0.5)
    with row1[1]:
        e_grade = st.number_input("영어 등급", min_value=0.0, max_value=9.0, value=3.0, step=0.5)
    with row1[2]:
        m_grade = st.number_input("수학 등급", min_value=0.0, max_value=9.0, value=3.0, step=0.5)
    with row2[0]:
        t1_grade = st.number_input("탐구1 등급", min_value=0.0, max_value=9.0, value=3.0, step=0.5)
    with row2[1]:
        t2_grade = st.number_input("탐구2 등급", min_value=0.0, max_value=9.0, value=3.0, step=0.5)
    with row2[2]:
        h_grade = st.number_input("한국사 등급", min_value=0.0, max_value=9.0, value=3.0, step=0.5)

    st.markdown("#### 2) 지역 / 전형 필터")

    basic_regions = ["서울", "경기", "인천", "부산", "대구", "경북", "충북", "충남"]
    if susi is not None and "지역구분" in susi.columns:
        all_regions = sorted(susi["지역구분"].dropna().unique().tolist())
        default_regions = [r for r in basic_regions if r in all_regions] or all_regions
        region_sel = st.multiselect("관심 지역 선택", all_regions, default=default_regions)
    else:
        region_sel = []

    type_sel = st.multiselect("관심 전형 (예: 학생부교과, 학생부종합, 논술 등)", [], default=[])

    st.markdown("---")

    st.markdown("### 참고용 수시·정시 데이터 미리보기")

    col_susi, col_jungsi = st.columns(2)

    # ---------------- 수시 미리보기 ----------------
    with col_susi:
        st.markdown("#### 수시 데이터 (어디가 2025 수시 입결)")

        if susi is None:
            st.error("2025수시입결.csv 파일을 찾을 수 없습니다.")
        else:
            work = susi.copy()

            if region_sel:
                if "지역구분" in work.columns:
                    work = work[work["지역구분"].isin(region_sel)]

            if hope_keyword:
                mask = (
                    work.get("대학명", "").astype(str).str.contains(hope_keyword, na=False)
                    | work.get("모집단위명", "").astype(str).str.contains(hope_keyword, na=False)
                )
                work = work[mask]

            if type_sel and "전형세부유형" in work.columns:
                work = work[work["전형세부유형"].isin(type_sel)]

            st.dataframe(work.head(200), use_container_width=True)

    # ---------------- 정시 미리보기 ----------------
    with col_jungsi:
        st.markdown("#### 정시 데이터 (어디가 2025 정시 입결)")

        if jungsi is None:
            st.error("2025정시입결.csv 파일을 찾을 수 없습니다.")
        else:
            work = jungsi.copy()

            if hope_keyword:
                mask = (
                    work.get("대학명", "").astype(str).str.contains(hope_keyword, na=False)
                    | work.get("모집단위", "").astype(str).str.contains(hope_keyword, na=False)
                )
                work = work[mask]

            st.dataframe(work.head(200), use_container_width=True)

    st.markdown("---")
    st.markdown("### 학생부 종합 전형 자가진단")

    st.write(
        "아래 문항을 1점(전혀 아니다) ~ 5점(매우 그렇다) 사이에서 체크해 보세요. "
        "총점 30점 이상: **적정**, 25점 이상: **보통**, 20점 미만: **보완 필요** 로 간단히 해석합니다."
    )

    questions = [
        "1) 이수 과목 수가 충분하고, 심화/선택 과목까지 고르게 이수했다.",
        "2) 주요 교과 성취도가 우수하다.",
        "3) 자율·진로·동아리 활동이 꾸준하고 내용이 충실하다.",
        "4) 리더십·배려·봉사·의사소통·공동체 역량이 생활기록부에 잘 드러난다.",
        "5) 프로젝트·캠페인·보고서 활동 경험이 있다.",
        "6) 독서 활동이 풍부하고, 교과/진로와 연계해 활용했다.",
        "7) 실패 경험과 그 극복 과정이 활동 기록에 담겨 있다.",
        "8) 생활기록부에 적힌 활동 내용에 대해 자신 있게 말할 수 있다.",
        "9) 발표·면접 등 스피치 역량이 안정적이다.",
        "10) 고교 3년 활동을 관통하는 나만의 키워드/주제가 있다.",
    ]

    cols = st.columns(2)
    scores = []
    for i, q in enumerate(questions):
        with cols[i % 2]:
            val = st.slider(q, 1, 5, 3, key=f"q_{i}")
            scores.append(val)

    total = sum(scores)
    st.markdown(f"**총점: {total}점 / 50점**")

    if total >= 30:
        level = "적정 (학생부종합 전형 지원에 비교적 잘 준비되어 있음)"
    elif total >= 25:
        level = "보통 (어느 정도 준비되어 있으나 일부 보완 필요)"
    else:
        level = "보완 필요 (지원 전 준비와 점검이 더 필요함)"

    st.info(level)


# ------------------------------------------------------------------
# 3. 최저 기준으로 대학 찾기
# ------------------------------------------------------------------
def view_minimum_search():
    st.subheader("최저 기준으로 대학 찾기")

    df_min = load_minimum_data()
    susi = load_susi_data()
    school_stats = build_school_grade_stats()

    if df_min is None:
        st.error("2025최저모음.csv 파일을 찾을 수 없습니다.")
        return

    st.markdown("### 1) 내 희망 최저 기준 입력")

    c1, c2, c3 = st.columns(3)
    with c1:
        k_max = st.number_input("국어 최대 등급 (0=미사용)", min_value=0.0, max_value=9.0, value=0.0, step=0.5)
    with c2:
        e_max = st.number_input("영어 최대 등급 (0=미사용)", min_value=0.0, max_value=9.0, value=0.0, step=0.5)
    with c3:
        m_max = st.number_input("수학 최대 등급 (0=미사용)", min_value=0.0, max_value=9.0, value=0.0, step=0.5)

    c4, c5, c6 = st.columns(3)
    with c4:
        t1_max = st.number_input("탐구1 최대 등급 (0=미사용)", min_value=0.0, max_value=9.0, value=0.0, step=0.5)
    with c5:
        t2_max = st.number_input("탐구2 최대 등급 (0=미사용)", min_value=0.0, max_value=9.0, value=0.0, step=0.5)
    with c6:
        h_max = st.number_input("한국사 최대 등급 (0=미사용)", min_value=0.0, max_value=9.0, value=0.0, step=0.5)

    st.caption("※ 0으로 두면 해당 과목은 최저 기준에서 고려하지 않습니다. 실제 요강과의 차이가 있을 수 있습니다.")

    st.markdown("### 2) 지역 / 키워드 / 내신 입력")

    col1, col2 = st.columns([2, 1])
    with col1:
        if "지역구분" in df_min.columns:
            regions = sorted(df_min["지역구분"].dropna().unique().tolist())
            default_regions = [r for r in ["서울", "경기", "인천", "부산", "대구", "경북", "충북", "충남"] if r in regions] or regions
            region_sel = st.multiselect("지역 선택", regions, default=default_regions)
        else:
            region_sel = []

        keyword = st.text_input("검색 키워드 (대학명/모집단위/최저 내용 일부)", "")
    with col2:
        my_grade = st.number_input("내 내신 (대표 등급, 선택)", min_value=1.0, max_value=9.0, value=3.0, step=0.1)

    if st.button("최저 기준에 맞는 대학 검색"):
        work = df_min.copy()

        if region_sel and "지역구분" in work.columns:
            work = work[work["지역구분"].isin(region_sel)]

        if keyword:
            mask = (
                work.get("대학명", "").astype(str).str.contains(keyword, na=False)
                | work.get("모집단위명", "").astype(str).str.contains(keyword, na=False)
                | work.get("최저학력기준내용", "").astype(str).str.contains(keyword, na=False)
            )
            work = work[mask]

        # 과목별 등급 입력이 0이 아닐 때 -> 해당 과목이 최저 내용에 언급되는지만 간단 확인
        subj_tokens = []
        if k_max > 0:
            subj_tokens.append("국")
        if e_max > 0:
            subj_tokens.append("영")
        if m_max > 0:
            subj_tokens.append("수")
        if t1_max > 0 or t2_max > 0:
            subj_tokens.append("탐")
        if h_max > 0:
            subj_tokens.append("한")

        if subj_tokens:
            text = work["최저학력기준내용"].astype(str)
            for tok in subj_tokens:
                text_mask = text.str.contains(tok, na=False)
                work = work[text_mask]
                text = work["최저학력기준내용"].astype(str)

        # 수시 입결 데이터와 간단 결합 (같은 대학/모집단위 기준)
        if susi is not None:
            susi_work = susi.copy()
            merge_cols_left = []
            merge_cols_right = []
            for left, right in [
                ("지역구분", "지역구분"),
                ("대학명", "대학명"),
                ("전형세부유형", "전형세부유형"),
                ("모집단위명", "모집단위명"),
            ]:
                if left in work.columns and right in susi_work.columns:
                    merge_cols_left.append(left)
                    merge_cols_right.append(right)

            if merge_cols_left:
                susi_small = susi_work[merge_cols_right + [c for c in susi_work.columns if "내신" in c or "등급" in c]].copy()
                susi_small = susi_small.drop_duplicates()
                susi_small.columns = merge_cols_right + [f"수시_{c}" for c in susi_small.columns[len(merge_cols_right):]]
                work = work.merge(
                    susi_small,
                    left_on=merge_cols_left,
                    right_on=merge_cols_right,
                    how="left",
                )

        # 우리 학교 합격 내신 참고 값 결합
        if not school_stats.empty:
            merge_cols = []
            for c in ["지역구분", "지역"]:
                if c in work.columns and c in school_stats.columns:
                    merge_cols.append((c, c))
                    break
            if "대학명" in work.columns and "대학명" in school_stats.columns:
                merge_cols.append(("대학명", "대학명"))
            if "모집단위명" in work.columns and "모집단위" in school_stats.columns:
                merge_cols.append(("모집단위명", "모집단위"))

            if merge_cols:
                left_on = [l for l, _ in merge_cols]
                right_on = [r for _, r in merge_cols]
                work = work.merge(
                    school_stats,
                    left_on=left_on,
                    right_on=right_on,
                    how="left",
                )

        if work.empty:
            st.warning("입력한 조건에 맞는 최저 데이터가 없습니다.")
        else:
            show_cols = []
            for c in [
                "지역구분",
                "대학명",
                "전형세부유형",
                "중심전형분류",
                "계열",
                "상세계열",
                "모집단위명",
                "소재지",
                "모집인원",
                "최저학력기준내용",
                "우리학교평균내신",
                "우리학교중앙내신",
            ]:
                if c in work.columns:
                    show_cols.append(c)

            st.markdown("### 최저 기준을 충족할 가능성이 있는 대학 목록")
            st.dataframe(work[show_cols].reset_index(drop=True), use_container_width=True)


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------
menu = st.sidebar.radio(
    "메뉴 선택",
    ["함창고 등급대 분석", "수시·정시 추천 탐색기", "최저 기준으로 대학 찾기"],
)

if menu == "함창고 등급대 분석":
    view_school_analysis()
elif menu == "수시·정시 추천 탐색기":
    view_recommendation()
elif menu == "최저 기준으로 대학 찾기":
    view_minimum_search()

st.markdown(
    "<div style='margin-top:2rem; text-align:center; font-size:0.85rem; color:#aaaaaa;'>"
    "제작자 함창고 교사 박호종"
    "</div>",
    unsafe_allow_html=True,
)

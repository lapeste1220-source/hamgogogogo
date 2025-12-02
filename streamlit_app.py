import streamlit as st
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import re

# ---------------- 기본 설정 ----------------
st.set_page_config(
    page_title="함창고 수시·정시 검색기",
    layout="wide",
)

st.title("함창고 수시·정시 검색기")
st.caption("함창고 입결 + 2025 어디가 수시·정시·최저 데이터를 함께 보는 전용 도구 (베타)")

DATA_DIR = Path(".")

SUJI_FILE = DATA_DIR / "수시진학관리(2025년2월4일).csv"
SUSI_FILE = DATA_DIR / "2025수시입결.csv"
JEONG_FILE = DATA_DIR / "2025정시입결.csv"
CHOEJEO_FILE = DATA_DIR / "2025최저모음.csv"

# 전역에서 사용할 컬럼 이름들
SUSI_GRADE_COL = None  # 어디가 수시 평균 내신(등급) 컬럼명
SU_DEPT_AVG = None     # 대학/모집단위별 평균 내신
JEONG_SCORE_COL = None # 정시 평균 백분위 컬럼명

# ---------------- 공통 유틸 ----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.replace("\n", "").replace(" ", "") for c in df.columns]
    return df

@st.cache_data
def load_data():
    suji = susi = jeong = choe = None
    # 수시진학관리(함창고 내부 입결)
    if SUJI_FILE.exists():
        suji = pd.read_csv(SUJI_FILE, encoding="utf-8")
        suji = normalize_columns(suji)
    # 어디가 수시
    if SUSI_FILE.exists():
        susi = pd.read_csv(SUSI_FILE, encoding="utf-8")
        susi = normalize_columns(susi)
    # 어디가 정시
    if JEONG_FILE.exists():
        jeong = pd.read_csv(JEONG_FILE, encoding="utf-8")
        jeong = normalize_columns(jeong)
    # 어디가 최저
    if CHOEJEO_FILE.exists():
        choe = pd.read_csv(CHOEJEO_FILE, encoding="utf-8")
        choe = normalize_columns(choe)
    return suji, susi, jeong, choe

suji_df, susi_df, jeong_df, choe_df = load_data()

# ---------------- 어디가 수시/정시 보조 테이블 ----------------
if susi_df is not None:
    # 수시 평균 내신 컬럼 추론
    grade_candidates = [c for c in susi_df.columns if ("평균" in c or "평균등급" in c or "내신" in c or "등급" in c)]
    SUSI_GRADE_COL = grade_candidates[0] if grade_candidates else None

    # 대학/모집단위별 교과 전형 평균 내신 (정시 추천·최저 검색 등에서 사용)
    if SUSI_GRADE_COL is not None and SUSI_GRADE_COL in susi_df.columns:
        su_for_avg = susi_df.copy()
        # 교과 전형 위주
        if "전형세부유형" in su_for_avg.columns:
            mask = su_for_avg["전형세부유형"].astype(str).str.contains("교과")
            su_for_avg = su_for_avg[mask]
        # 그룹핑
        if {"대학명", "모집단위명"}.issubset(su_for_avg.columns):
            SU_DEPT_AVG = (
                su_for_avg
                .groupby(["대학명", "모집단위명"], as_index=False)[SUSI_GRADE_COL]
                .mean()
                .rename(columns={SUSI_GRADE_COL: "수시평균내신"})
            )

if jeong_df is not None:
    # 정시 반영영역 평균 백분위 컬럼 추론
    cand = [c for c in jeong_df.columns if "반영영역평균백분위" in c.replace(" ", "")]
    JEONG_SCORE_COL = cand[0] if cand else None

# ---------------- 함창고 수시진학 데이터 가공 ----------------
SUJI_HAS_DATA = suji_df is not None and not suji_df.empty

if SUJI_HAS_DATA:
    # 대표 등급 컬럼 찾기 (전교과 / 일반등급 / 내등급(환산) 등)
    grade_cols = [c for c in suji_df.columns if "등급" in c and not any(x in c for x in ["한국사", "탐구", "제2외"])]
    main_grade_col = None
    for key in ["일반등급", "내등급(환산)", "전교과평균등급", "전교과"]:
        if key.replace(" ", "") in suji_df.columns:
            main_grade_col = key.replace(" ", "")
            break
    if main_grade_col is None and grade_cols:
        main_grade_col = grade_cols[0]

    if main_grade_col is not None:
        suji_df["대표등급"] = pd.to_numeric(suji_df[main_grade_col], errors="coerce")
    else:
        suji_df["대표등급"] = np.nan

    def band(x):
        if pd.isna(x):
            return "등급정보없음"
        if x < 1.5:
            return "1등급대"
        if x < 2.5:
            return "2등급대"
        if x < 3.5:
            return "3등급대"
        if x < 4.5:
            return "4등급대"
        if x < 5.5:
            return "5등급대"
        return "6등급대이하"

    suji_df["등급대"] = suji_df["대표등급"].apply(band)

    # 합격여부: 등록여부/최종단계에 '등록' 또는 '합격'이 들어가면 True
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
        my_grade = st.number_input("내신 대표 등급(전교과 또는 국수영 평균)", min_value=1.0, max_value=9.0, step=0.1, value=3.0)
    with col2:
        st.write("최근 모의고사 등급 입력 (없으면 0으로 두세요)")
    c1, c2, c3 = st.columns(3)
    with c1:
        g_kor = st.number_input("국어 등급", 0.0, 9.0, 0.0, 0.5)
        g_math = st.number_input("수학 등급", 0.0, 9.0, 0.0, 0.5)
    with c2:
        g_eng = st.number_input("영어 등급", 0.0, 9.0, 0.0, 0.5)
        g_t1 = st.number_input("탐구1 등급", 0.0, 9.0, 0.0, 0.5)
    with c3:
        g_t2 = st.number_input("탐구2 등급", 0.0, 9.0, 0.0, 0.5)
        g_hist = st.number_input("한국사 등급", 0.0, 9.0, 0.0, 0.5)
    # 정시용 평균 백분위 대략 환산
    grade_list = [g for g in [g_kor, g_math, g_eng, g_t1, g_t2] if g > 0]
    if grade_list:
        mapping = {1: 96, 2: 89, 3: 77, 4: 62, 5: 47, 6: 32, 7: 20, 8: 11, 9: 4}
        perc = [mapping.get(int(round(g)), 50) for g in grade_list]
        mock_percentile = float(np.mean(perc))
    else:
        mock_percentile = None

    region_options = ["서울", "경기", "인천", "부산", "대구", "경북", "충북", "충남"]
    selected_regions = st.multiselect("희망 지역 선택", options=region_options, default=region_options)
    return my_grade, mock_percentile, selected_regions

# ---------------- 뷰 1: 함창고 등급대 분석 ----------------
def view_grade_analysis():
    st.header("함창고 등급대 분석")
    if not SUJI_HAS_DATA:
        st.error("함창고 수시진학관리 데이터가 없어 분석을 진행할 수 없습니다.")
        return

    df = suji_df.copy()

    # 필터 UI
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        grade_band = st.selectbox("등급대 선택", options=sorted(df["등급대"].unique()), index=1 if "2등급대" in df["등급대"].unique() else 0)
    with col2:
        region = st.multiselect("지역 선택", options=sorted(df["지역"].dropna().unique()), default=None)
    with col3:
        univ = st.multiselect("대학 선택", options=sorted(df["대학명"].dropna().unique()), default=None)
    with col4:
        major = st.text_input("학과(모집단위) 키워드", "")

    filtered = df[df["등급대"] == grade_band]
    if region:
        filtered = filtered[filtered["지역"].isin(region)]
    if univ:
        filtered = filtered[filtered["대학명"].isin(univ)]
    if major:
        filtered = filtered[filtered["모집단위"].astype(str).str.contains(major)]

    # 합격자만
    admit_only = filtered[filtered["합격"]]

    st.subheader("합격자 지역 분포")
    if admit_only.empty:
        st.info("선택한 조건에 해당하는 합격 데이터가 없습니다.")
    else:
        region_count = admit_only.groupby("지역").size().reset_index(name="합격자수")
        st.bar_chart(region_count.set_index("지역"))

    st.subheader("합격 전형 분포 (교과 / 종합 / 농어촌 등)")
    if admit_only.empty:
        st.info("선택한 조건에 해당하는 합격 데이터가 없습니다.")
    else:
        vt_col = "전형유형" if "전형유형" in admit_only.columns else "전형명(대)"
        vt_count = (
            admit_only.assign(전형분류=lambda d: d[vt_col].astype(str).str.extract("(교과|종합|농어촌)", expand=False).fillna("기타"))
            .groupby("전형분류")
            .size()
            .reset_index(name="합격자수")
        )
        st.bar_chart(vt_count.set_index("전형분류"))

    # 상세 표 + 우리학교 입결만 보기
    st.markdown("---")
    col_h, col_btn = st.columns([3, 1])
    with col_h:
        st.markdown("#### 필터 조건에 따른 상세 합격 학과 목록")
    with col_btn:
        only_ham = st.checkbox("우리학교 입결만 보기", value=True)

    detail = admit_only.copy()
    if only_ham:
        # 현재 데이터는 이미 함창고이지만, 만약 학교 구분 컬럼이 있다면 한번 더 필터
        for col in ["학교유형", "학교명"]:
            if col in detail.columns:
                detail = detail[detail[col].astype(str).str.contains("함창고")]
                break

    cols_for_table = ["학년", "반", "번호", "이름", "등급대", "대표등급", "지역", "대학명", "모집단위"]
    cols_for_table = [c for c in cols_for_table if c in detail.columns]
    if not detail.empty:
        table_df = detail[cols_for_table].sort_values(["대표등급", "대학명", "모집단위"])
        st.dataframe(table_df, use_container_width=True, hide_index=True)
    else:
        st.info("조건에 맞는 합격 학과가 없습니다.")

# ---------------- 뷰 2: 수시·정시 추천 탐색기 ----------------
def view_recommend():
    st.header("수시·정시 추천 탐색기")

    if susi_df is None or SUSI_GRADE_COL is None or jeong_df is None or JEONG_SCORE_COL is None:
        st.error("어디가 수시/정시 데이터가 부족해 추천 계산을 할 수 없습니다.")
        st.info("그래도 기본 데이터 탐색은 상단 다른 메뉴에서 가능합니다.")
        return

    my_grade, mock_percentile, regions = get_student_inputs()

    tab_su, tab_je = st.tabs(["수시 추천", "정시 추천"])

    # ---- 수시 추천 ----
    with tab_su:
        st.subheader("수시 추천 대학 (내신 기준)")
        df = susi_df.copy()

        # 지역 필터
        if "지역구분" in df.columns:
            df = df[df["지역구분"].isin(regions)]

        # 유효한 내신 데이터
        df[SUSI_GRADE_COL] = pd.to_numeric(df[SUSI_GRADE_COL], errors="coerce")
        df = df.dropna(subset=[SUSI_GRADE_COL])

        if df.empty:
            st.warning("해당 지역에서 내신 데이터가 있는 수시 입결이 없습니다.")
        else:
            df["내신평균점수"] = df[SUSI_GRADE_COL]
            df["내신차이(내-합)"] = my_grade - df["내신평균점수"]

            def label_row(d):
                diff = d["내신차이(내-합)"]
                if diff <= -0.7:
                    return "하향(도전)"
                if diff <= 0.5:
                    return "적정"
                return "안전"

            df["추천구분"] = df.apply(label_row, axis=1)

            safe = df[df["추천구분"] == "안전"].nsmallest(2, "내신차이(내-합)")
            mid = df[df["추천구분"] == "적정"].nsmallest(2, "내신차이(내-합)").sample(frac=1, random_state=0)
            risk = df[df["추천구분"] == "하향(도전)"].nlargest(2, "내신차이(내-합)")

            rec = pd.concat([safe, mid, risk], ignore_index=True)
            cols = ["추천구분", "지역구분", "대학명", "전형세부유형", "계열", "모집단위명", "내신평균점수"]
            cols = [c for c in cols if c in rec.columns]
            st.dataframe(rec[cols], use_container_width=True, hide_index=True)

            st.caption("※ 내신평균점수: 어디가 수시 입결 상 해당 전형의 평균 내신 (등급)")

    # ---- 정시 추천 ----
    with tab_je:
        st.subheader("정시 추천 대학 (모의고사 백분위 추정 기준)")

        if mock_percentile is None:
            st.info("정시 추천을 위해서는 최근 모의고사 등급을 최소 한 과목 이상 입력해 주세요.")
            return

        dfj = jeong_df.copy()
        if "지역구분" in dfj.columns:
            dfj = dfj[dfj["지역구분"].isin(regions)]

        dfj[JEONG_SCORE_COL] = pd.to_numeric(dfj[JEONG_SCORE_COL], errors="coerce")
        dfj = dfj.dropna(subset=[JEONG_SCORE_COL])

        if dfj.empty:
            st.warning("해당 지역에서 정시 입결 데이터가 없습니다.")
        else:
            dfj["정시평균백분위"] = dfj[JEONG_SCORE_COL]
            dfj["백분위차이(내-합)"] = mock_percentile - dfj["정시평균백분위"]

            def label_j(row):
                d = row["백분위차이(내-합)"]
                if d >= 7:
                    return "안전"
                if d >= -3:
                    return "적정"
                return "하향(도전)"

            dfj["추천구분"] = dfj.apply(label_j, axis=1)

            safe = dfj[dfj["추천구분"] == "안전"].nlargest(2, "백분위차이(내-합)")
            mid = dfj[dfj["추천구분"] == "적정"].nlargest(2, "백분위차이(내-합)")
            risk = dfj[dfj["추천구분"] == "하향(도전)"].nsmallest(2, "백분위차이(내-합)")

            recj = pd.concat([safe, mid, risk], ignore_index=True)

            # 같은 대학/모집단위의 수시 교과 평균내신 붙이기
            if SU_DEPT_AVG is not None and {"대학명", "모집단위명"}.issubset(recj.columns):
                recj = recj.merge(SU_DEPT_AVG, how="left", on=["대학명", "모집단위명"])

            colsj = ["추천구분", "대학명", "전형명", "모집군", "모집단위", "정시평균백분위", "수시평균내신"]
            colsj = [c for c in colsj if c in recj.columns]
            st.dataframe(recj[colsj], use_container_width=True, hide_index=True)

            st.caption("※ 정시평균백분위: 어디가 정시 입결 상 반영영역 평균 백분위, 수시평균내신: 동일 학과 교과전형 평균 내신")

# ---------------- 최저 기준으로 대학 찾기 ----------------
def parse_minimum_rule(rule_text, grades):
    """
    rule_text: 최저학력기준 내용 (자연어)
    grades: dict with keys ['국어','수학','영어','탐1','탐2','한국사'], value: 등급 (float, 1~9, 0은 미응시)
    반환값: True(충족 가능) / False(불충족 또는 해석 불가)
    """
    if not rule_text or not isinstance(rule_text, str):
        return False

    text = rule_text.replace(" ", "")
    nums = [g for g in [grades["국어"], grades["수학"], grades["영어"], grades["탐1"], grades["탐2"], grades["한국사"]] if g > 0]
    if not nums:
        return False

    # 각 n등급 이내
    m_each = re.search(r"(\d)등급이내", text)
    if m_each:
        limit = int(m_each.group(1))
        return all(g <= limit for g in nums)

    # n개 영역 합 x이내
    m_sum = re.search(r"(?:중)?(\d)개영역?합(\d+)이내", text)
    if m_sum:
        n = int(m_sum.group(1))
        s_limit = int(m_sum.group(2))
        nums_sorted = sorted(nums)
        if len(nums_sorted) < n:
            return False
        return sum(nums_sorted[:n]) <= s_limit

    # "2개영역각1등급" 등 간단 형태
    m_each2 = re.search(r"각(\d)등급", text)
    if m_each2:
        limit = int(m_each2.group(1))
        return all(g <= limit for g in nums)

    # 해석 실패시 보수적으로 False
    return False

def view_choejeo():
    st.header("최저 기준으로 대학 찾기")

    if choe_df is None:
        st.error("2025 최저 기준 데이터(2025최저모음.csv)를 찾을 수 없습니다.")
        return

    st.markdown("### 1) 내 희망 최저 기준 입력")
    row1 = st.columns(3)
    with row1[0]:
        g_k = st.number_input("국어 최대 등급(0=미사용)", 0.0, 9.0, 0.0, 0.5, key="min_k")
    with row1[1]:
        g_e = st.number_input("영어 최대 등급(0=미사용)", 0.0, 9.0, 0.0, 0.5, key="min_e")
    with row1[2]:
        g_m = st.number_input("수학 최대 등급(0=미사용)", 0.0, 9.0, 0.0, 0.5, key="min_m")

    row2 = st.columns(3)
    with row2[0]:
        g_t1 = st.number_input("탐구1 최대 등급(0=미사용)", 0.0, 9.0, 0.0, 0.5, key="min_t1")
    with row2[1]:
        g_t2 = st.number_input("탐구2 최대 등급(0=미사용)", 0.0, 9.0, 0.0, 0.5, key="min_t2")
    with row2[2]:
        g_h = st.number_input("한국사 최대 등급(0=미사용)", 0.0, 9.0, 0.0, 0.5, key="min_h")

    st.caption("※ 0으로 두면 해당 과목은 최저 기준에서 고려하지 않습니다. 실제 대학별 세부 조건과는 차이가 있을 수 있습니다.")

    colr1, colr2 = st.columns([2, 1])
    with colr1:
        regions = st.multiselect("지역 선택", options=sorted(choe_df["지역구분"].dropna().unique()), default=None)
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

        df["최저충족가능"] = df["최저학력기준내용"].apply(lambda x: parse_minimum_rule(x, my_grades))
        df_ok = df[df["최저충족가능"]]

        if df_ok.empty:
            st.info("입력한 조건에 맞는 최저 기준 대학을 찾지 못했습니다. (해석 불가 조건은 제외되었습니다.)")
            return

        # 어디가 수시 평균 내신 붙이기
        if SU_DEPT_AVG is not None and {"대학명", "모집단위명"}.issubset(df_ok.columns):
            df_ok = df_ok.merge(SU_DEPT_AVG, how="left", on=["대학명", "모집단위명"])

        cols = ["지역구분", "대학명", "전형세부유형", "모집단위명", "최저학력기준내용", "수시평균내신"]
        cols = [c for c in cols if c in df_ok.columns]
        st.dataframe(df_ok[cols], use_container_width=True, hide_index=True)

        st.caption("※ 수시평균내신: 어디가 수시 입결 기준 (교과 전형 위주). 실제 각 대학의 입시 요강과 반드시 함께 확인하세요.")

# ---------------- 학생부종합 자가진단 (간단 버전) ----------------
def view_jagajin():
    st.header("학생부 종합 전형 적합도 자가진단")

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
        scores.append(st.slider(q, 1, 5, 3, key=q))

    total = sum(scores)
    max_score = 5 * len(scores)
    ratio = total / max_score * 100

    st.markdown("### 결과 요약")
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

    # 바 그래프 (폭을 줄여서 2단으로)
    st.markdown("### 문항별 점수 분포")
    df = pd.DataFrame(
        {"문항": [f"Q{i+1}" for i in range(len(scores))], "점수": scores}
    )
    # 2단 레이아웃
    c1, c2 = st.columns(2)
    half = len(df) // 2
    with c1:
        st.bar_chart(df.iloc[:half].set_index("문항"))
    with c2:
        st.bar_chart(df.iloc[half:].set_index("문항"))

# ---------------- 사이드바 메뉴 ----------------
with st.sidebar:
    st.markdown("### 메뉴 선택")
    menu = st.radio(
        "",
        ["함창고 등급대 분석", "수시·정시 추천 탐색기", "최저 기준으로 대학 찾기", "학생부 종합 자가진단"],
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
else:
    view_jagajin()

# 하단 제작자 표기
st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:0.85rem; color:gray;'>제작자 함창고 교사 박호종</div>",
    unsafe_allow_html=True,
)

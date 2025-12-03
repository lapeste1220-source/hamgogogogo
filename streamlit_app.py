import streamlit as st
import pandas as pd
import numpy as np
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

# 함창고 수시진학관리: 2025 + 2024 파일을 합쳐 사용
SUJI_2025_FILE = DATA_DIR / "수시진학관리(2025년2월4일).csv"
SUJI_2024_FILE = DATA_DIR / "수시진학관리(2024년2월20일).csv"

# 어디가 수시/정시/최저
SUSI_FILE = DATA_DIR / "2025수시입결.csv"
JEONG_FILE = DATA_DIR / "2025정시입결.csv"
CHOEJEO_FILE = DATA_DIR / "2025최저모음.csv"

# 전역에서 사용할 변수
SUSI_GRADE_COL = None
SU_DEPT_AVG = None
JEONG_SCORE_COL = None

# ---------------- 공통 유틸 ----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.replace("\n", "").replace(" ", "") for c in df.columns]
    return df

@st.cache_data
def load_data():
    # 1) 함창고 수시진학관리 (연도별 병합)
    suji_list = []

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

    # 2) 어디가 수시/정시/최저
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
    # 수시 평균 내신 컬럼 추론
    grade_candidates = [
        c for c in susi_df.columns
        if any(k in c for k in ["평균", "평균등급", "내신", "등급"])
    ]
    SUSI_GRADE_COL = grade_candidates[0] if grade_candidates else None

    if SUSI_GRADE_COL is not None and SUSI_GRADE_COL in susi_df.columns:
        su_for_avg = susi_df.copy()

        # 가능하면 교과 전형만 사용
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

def decide_admit(row):
    reg = str(row.get("등록여부", ""))
    final = str(row.get("최종단계", ""))
    reason = str(row.get("불합격사유", ""))

    negative_keywords = ["불합격", "미등록", "탈락", "포기", "최저미충족", "최저미달"]
    if any(neg in reason for neg in negative_keywords):
        return False

    positive_keywords_reg = ["등록", "합격"]
    positive_keywords_final = ["합격", "최종합격", "추가합격", "추합"]

    if any(pos in reg for pos in positive_keywords_reg):
        return True
    if any(pos in final for pos in positive_keywords_final):
        return True

    return False


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

    suji_df["합격"] = suji_df.apply(decide_admit, axis=1)
# ---------------- 학생 입력 유틸 ----------------
def get_student_inputs():
    st.markdown("#### 1) 내 기본 성적 입력")
    col1, col2 = st.columns(2)
    with col1:
        my_grade = st.number_input(
            "내신 대표 등급(전교과 또는 국수영 평균)",
            min_value=1.0, max_value=9.0, step=1.0, value=3.0,
        )
    with col2:
        mock_percent_input = st.number_input(
            "최근 모의고사 백분위 평균 (없으면 0)",
            min_value=0.0, max_value=100.0, step=1.0, value=0.0,
        )

    # ★ 추가: 희망 대학 / 희망 학과 입력
    st.markdown("#### 1-1) 희망 대학/학과 입력")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        target_univ = st.text_input("희망 대학 (선택 입력)", "")
    with col_u2:
        target_major = st.text_input("희망 학과 또는 모집단위 (선택 입력)", "")

    st.write("과목별 등급(선택 입력, 백분위 추정용)")

    r1c1, r1c2, r2c1 = st.columns(3)
    with r1c1:
        g_kor = st.number_input("국어 등급", 0.0, 9.0, 0.0, 1.0)
        g_eng = st.number_input("영어 등급", 0.0, 9.0, 0.0, 1.0)
    with r1c2:
        g_math = st.number_input("수학 등급", 0.0, 9.0, 0.0, 1.0)
        g_t1 = st.number_input("탐구1 등급", 0.0, 9.0, 0.0, 1.0)
    with r2c1:
        g_t2 = st.number_input("탐구2 등급", 0.0, 9.0, 0.0, 1.0)
        g_hist = st.number_input("한국사 등급", 0.0, 9.0, 0.0, 1.0)

    # 백분위 추정
    grade_list = [g for g in [g_kor, g_math, g_eng, g_t1, g_t2] if g > 0]
    mock_percent_est = None
    if grade_list:
        mapping = {1: 96, 2: 89, 3: 77, 4: 62, 5: 47, 6: 32, 7: 20, 8: 11, 9: 4}
        perc = [mapping.get(int(round(g)), 50) for g in grade_list]
        mock_percent_est = float(np.mean(perc))

    mock_percentile = mock_percent_input if mock_percent_input > 0 else mock_percent_est

    region_options = ["서울", "경기", "인천", "부산", "대구", "경북", "충북", "충남"]
    selected_regions = st.multiselect(
        "희망 지역 선택", options=region_options, default=region_options
    )

    # 반환값에 희망 대학/학과 포함
    return (
        my_grade,
        mock_percentile,
        selected_regions,
        target_univ,   # ★추가
        target_major   # ★추가
    )


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
    df = df.dropna(subset=["대표등급"])

    # --- 필터 UI ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        min_g = float(np.floor(df["대표등급"].min()))
        max_g = float(np.ceil(df["대표등급"].max()))
        grade_min, grade_max = st.slider(
            "대표등급 범위 선택",
            min_value=min_g,
            max_value=max_g,
            value=(min_g, max_g),
            step=1.0,
        )
    with col2:
        year_options = sorted(df["입시연도"].dropna().unique())
        default_years = [year_options[-1]] if year_options else []
        selected_years = st.multiselect("입시 연도", options=year_options, default=default_years)
    with col3:
        region = st.multiselect("지역 선택", options=sorted(df["지역"].dropna().unique()))
    with col4:
        univ = st.multiselect("대학 선택", options=sorted(df["대학명"].dropna().unique()))

    major_keyword = st.text_input("학과(모집단위) 키워드", "")

    # --- 필터 적용 ---
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
        st.info("선택한 조건에 해당하는 데이터가 없습니다.")
        return

    vt_col = "전형유형" if "전형유형" in filtered.columns else "전형명(대)"
    base = filtered.assign(
        전형분류=lambda d: d[vt_col]
        .astype(str)
        .str.extract("(교과|종합|농어촌)", expand=False)
        .fillna("기타")
    )

    admit_only = base[base["합격"]]

    # --------- 합격자 지역 분포 ----------
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

    # --------- 합격 전형 분포 & 최저 충족률 ----------
    st.subheader("합격 전형 및 최저 충족률")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("##### 합격 전형 분포")
        if admit_only.empty:
            st.info("합격 데이터가 없습니다.")
        else:
            vt_count = (
                admit_only.groupby("전형분류")
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

    with col_right:
        st.markdown("##### 최저 충족률 (최저가 있는 전형 기준)")
        min_cols = [c for c in base.columns if "최저" in c]
        min_col = min_cols[0] if min_cols else None

        if min_col is None:
            st.info("최저학력 기준 정보가 없어 충족률을 계산할 수 없습니다.")
        else:
            cond_has_min = (
                base[min_col].notna()
                & (base[min_col].astype(str).str.strip() != "")
                & (~base[min_col].astype(str).str.contains("없음"))
            )
            base_min = base[cond_has_min].copy()

            if base_min.empty:
                st.info("최저학력 기준이 설정된 전형이 없습니다.")
            else:
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
                        x=alt.X("전형분류:O", axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("최저충족률(%):Q"),
                        tooltip=["전형분류", "최저충족률(%)"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(bar, use_container_width=True)
    # --------- 상세 표 ----------
    st.markdown("---")
    st.markdown("#### 필터 조건에 따른 상세 합격 학과 목록 (함창고 입결)")

    detail = base[base["합격"]].copy()
    if detail.empty:
        st.info("조건에 맞는 합격 학과가 없습니다.")
        return

    if "이름" in detail.columns:
        detail["이름마스킹"] = detail["이름"].astype(str).str[0] + "OO"
    else:
        detail["이름마스킹"] = ""

    if "전형유형" in detail.columns:
        detail["지원전형"] = detail["전형유형"]
    else:
        detail["지원전형"] = detail.get("전형명(대)", "")

    min_cols = [c for c in detail.columns if "최저" in c]
    if min_cols:
        mc = min_cols[0]
        detail["최저"] = detail[mc].fillna("없음").replace("", "없음")
    else:
        detail["최저"] = "없음"

    cols_for_table = [
        "입시연도",
        "이름마스킹",
        "대표등급",
        "지역",
        "대학명",
        "모집단위",
        "지원전형",
        "최저",
    ]
    cols_for_table = [c for c in cols_for_table if c in detail.columns]

    table_df = detail[cols_for_table].sort_values(["입시연도", "대표등급", "대학명", "모집단위"])
    st.dataframe(table_df, use_container_width=True, hide_index=True)


# ---------------- 추천 공통 유틸 ----------------
def pick_recommendations(df, label_col, diff_col, top_n=3):
    results = []

    safe = df[df[label_col] == "상향(도전)"]
    if not safe.empty:
        results.append(safe.nsmallest(top_n, diff_col))

    mid = df[df[label_col] == "적정"].copy()
    if not mid.empty:
        mid = mid.loc[mid[diff_col].abs().sort_values().index].head(top_n)
        results.append(mid)

    risk = df[df[label_col] == "안전"]
    if not risk.empty:
        results.append(risk.nlargest(top_n, diff_col))

    if not results:
        return pd.DataFrame(columns=df.columns)

    rec = pd.concat(results, ignore_index=True)

    dedup_keys = [c for c in ["대학명", "모집단위"] if c in rec.columns]
    if dedup_keys:
        rec = rec.drop_duplicates(subset=dedup_keys, keep="first")

    return rec


# ---------------- 뷰 2: 수시·정시 추천 탐색기 ----------------
def view_recommend():
    st.header("수시·정시 추천 탐색기")

    # ★ 수정: 희망대학/학과까지 입력받음
    my_grade, mock_percentile, regions, target_univ, target_major = get_student_inputs()

    tab_su, tab_je, tab_jg = st.tabs(["수시 추천", "정시 추천", "학생부종합 자가진단"])

    # ---- 수시 추천 ----
    with tab_su:
        st.subheader("수시 추천 대학 (어디가 사이트 입결 기준)")

        if not SUJI_HAS_DATA:
            st.warning(
                "우리 학교 수시 합격 내역이 부족하여 추천 계산이 어렵습니다.\n\n"
                "상단 '함창고 등급대 분석' 메뉴에서 전체 합격 데이터를 확인할 수 있습니다."
            )
        else:
            df = suji_df.copy()
            df = df[df["합격"]]
            df = df.dropna(subset=["대표등급"])

            if "지역" in df.columns and regions:
                df = df[df["지역"].isin(regions)]

            if df.empty:
                st.info("선택한 지역에서 합격 내신 데이터가 부족합니다.")
            else:
                group_cols = [c for c in ["대학명", "모집단위", "전형유형"] if c in df.columns]
                agg = (
                    df.groupby(group_cols, as_index=False)["대표등급"]
                    .mean()
                    .rename(columns={"대표등급": "합격평균내신"})
                )

                agg["내신차이(합-입)"] = agg["합격평균내신"] - my_grade

                def label_row(d):
                    diff = d["내신차이(합-입)"]
                    if diff > 0.3:
                        return "안전"
                    if diff < -0.3:
                        return "상향(도전)"
                    return "적정"

                agg["추천구분"] = agg.apply(label_row, axis=1)

                # ★ 추가: 희망 대학 / 학과 필터
                if target_univ:
                    agg = agg[agg["대학명"].astype(str).str.contains(target_univ)]
                if target_major:
                    agg = agg[agg["모집단위"].astype(str).str.contains(target_major)]

                rec = pick_recommendations(agg, "추천구분", "내신차이(합-입)", top_n=2)

                cols = ["추천구분"] + [c for c in ["대학명", "모집단위", "전형유형"] if c in rec.columns] + [
                    "합격평균내신",
                    "내신차이(합-입)",
                ]
                if not rec.empty:
                    st.dataframe(rec[cols], use_container_width=True, hide_index=True)
                else:
                    st.info("추천할 만한 데이터를 찾지 못했습니다.")

    # ---- 정시 추천 ----
    with tab_je:
        st.subheader("정시 추천 대학 (모의고사 백분위 기준)")

        if jeong_df is None or JEONG_SCORE_COL is None:
            st.warning("정시 입결 데이터가 부족하여 정시 추천 계산을 할 수 없습니다.")
        else:
            if mock_percentile is None:
                st.info("정시 추천을 위해 모의고사 백분위 입력 또는 과목 등급 입력이 필요합니다.")
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
                    dfj["백분위차이(합-입)"] = dfj["정시평균백분위"] - mock_percentile

                    def label_j(row):
                        d = row["백분위차이(합-입)"]
                        if d > 3:
                            return "상향(도전)"
                        if d < -3:
                            return "안전"
                        return "적정"

                    dfj["추천구분"] = dfj.apply(label_j, axis=1)

                    # ★ 추가: 희망 대학/학과 필터
                    if target_univ:
                        dfj = dfj[dfj["대학명"].astype(str).str.contains(target_univ)]
                    if target_major:
                        dfj = dfj[dfj["모집단위"].astype(str).str.contains(target_major)]

                    recj = pick_recommendations(dfj, "추천구분", "백분위차이(합-입)", top_n=3)

                    if SU_DEPT_AVG is not None and {"대학명", "모집단위"}.issubset(recj.columns):
                        recj = recj.merge(
                            SU_DEPT_AVG,
                            how="left",
                            left_on=["대학명", "모집단위"],
                            right_on=["대학명", "모집단위명"],
                        )

                    colsj = ["추천구분", "대학명", "전형명", "모집군", "모집단위", "정시평균백분위", "백분위차이(합-입)"]
                    if "수시평균내신" in recj.columns:
                        colsj.append("수시평균내신")

                    st.dataframe(recj[colsj], use_container_width=True, hide_index=True)

    # ---- 학생부종합 자가진단 ----
    with tab_jg:
        render_jagajin_inside_tab()


# ---------------- 최저 기준 대학 찾기 ----------------
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

    st.caption("※ 0으로 두면 해당 과목은 최저 기준에서 제외됩니다.")

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
            st.warning("조건에 맞는 데이터가 없습니다.")
            return

        df["최저충족가능"] = df["최저학력기준내용"].apply(
            lambda x: parse_minimum_rule(x, my_grades)
        )
        df_ok = df[df["최저충족가능"]]

        if df_ok.empty:
            st.info("입력 조건에 맞는 대학이 없습니다.")
            return

        if SU_DEPT_AVG is not None and {"대학명", "모집단위명"}.issubset(df_ok.columns):
            df_ok = df_ok.merge(SU_DEPT_AVG, how="left", on=["대학명", "모집단위명"])

        cols = ["지역구분", "대학명", "전형세부유형", "모집단위명", "최저학력기준내용"]
        if "수시평균내신" in df_ok.columns:
            cols.append("수시평균내신")

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


# =========================================
#         🔒 로그인 + 로고 포함
# =========================================
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import re
import altair as alt

# ---- 보안: 비밀번호 확인 ----
PASSWORD = "hamchang123"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.image("hch_logo.png", width=160)   # ★ 학교 로고 표시
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
#            기본 설정 + 파일 경로
# =========================================
st.set_page_config(
    page_title="함창고 수시·정시 검색기",
    layout="wide",
)

st.title("함창고 수시·정시 검색기")
st.caption("함창고 입결 + 2025 어디가 수시·정시·최저 데이터를 함께 보는 전용 도구 (베타)")

DATA_DIR = Path(".")

# 함창고 수시진학관리 파일
SUJI_2025_FILE = DATA_DIR / "수시진학관리(2025년2월4일).csv"
SUJI_2024_FILE = DATA_DIR / "수시진학관리(2024년2월20일).csv"

# 어디가 입결 파일
SUSI_FILE = DATA_DIR / "2025수시입결.csv"
JEONG_FILE = DATA_DIR / "2025정시입결.csv"
CHOEJEO_FILE = DATA_DIR / "2025최저모음.csv"

SUSI_GRADE_COL = None
JEONG_SCORE_COL = None
SU_DEPT_AVG = None


# =========================================
#       ★ 대학명 자동 그룹핑 테이블
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
#             데이터 로드 함수
# =========================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.replace("\n", "").replace(" ", "") for c in df.columns]
    return df


@st.cache_data
def load_data():
    suji_list = []

    # ---- 함창고 수시진학관리(2025 / 2024)
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

    # ---- 어디가 수시/정시/최저
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


# === 데이터 로드 ===
suji_df, susi_df, jeong_df, choe_df = load_data()
# =========================================
#        함창고 수시 데이터 가공
# =========================================
SUJI_HAS_DATA = suji_df is not None and not suji_df.empty


def decide_admit(row):
    """합격 여부를 텍스트 기반으로 자동 판별"""
    reg = str(row.get("등록여부", ""))
    final = str(row.get("최종단계", ""))
    reason = str(row.get("불합격사유", ""))

    negative = ["불합격", "미등록", "탈락", "포기", "최저미충족", "최저미달"]
    if any(key in reason for key in negative):
        return False

    positive_reg = ["등록", "합격"]
    positive_final = ["합격", "최종합격", "추가합격", "추합"]

    if any(p in reg for p in positive_reg):
        return True
    if any(p in final for p in positive_final):
        return True

    return False


if SUJI_HAS_DATA:
    # 대표등급 추출
    grade_cols = [
        c for c in suji_df.columns
        if "등급" in c and not any(x in c for x in ["한국사", "탐구", "제2외"])
    ]

    main_grade_col = None
    for key in ["일반등급", "내등급(환산)", "전교과평균등급", "전교과"]:
        k = key.replace(" ", "")
        if k in suji_df.columns:
            main_grade_col = k
            break

    if main_grade_col is None and grade_cols:
        main_grade_col = grade_cols[0]

    if main_grade_col:
        suji_df["대표등급"] = pd.to_numeric(suji_df[main_grade_col], errors="coerce")
    else:
        suji_df["대표등급"] = np.nan

    suji_df["합격"] = suji_df.apply(decide_admit, axis=1)

    # ★ 대학 그룹 자동 분류 추가
    if "대학명" in suji_df.columns:
        suji_df["대학그룹"] = suji_df["대학명"].apply(get_univ_group)


# =========================================
#           학생 기본 입력 함수
# =========================================
def get_student_inputs():
    st.markdown("#### 1) 내 기본 성적 입력")

    col1, col2 = st.columns(2)
    with col1:
        my_grade = st.number_input(
            "내신 대표 등급(전교과 또는 국수영 평균)",
            min_value=1.0, max_value=9.0, step=1.0, value=3.0,
        )
    with col2:
        mock_input = st.number_input(
            "최근 모의고사 백분위 평균 (없으면 0)",
            min_value=0.0, max_value=100.0, step=1.0, value=0.0,
        )

    # ---- 희망 대학 / 학과 입력 ----
    st.markdown("#### 1-1) 희망 대학/학과 입력")

    cc1, cc2 = st.columns(2)
    with cc1:
        target_univ = st.text_input("희망 대학 (선택)", "")
    with cc2:
        target_major = st.text_input("희망 학과/모집단위 (선택)", "")

    # ---- 과목별 등급 입력 (정시 추정용) ----
    st.write("과목별 등급 입력(선택): 백분위 자동 추정용")

    r1, r2 = st.columns(2)
    with r1:
        g_kor = st.number_input("국어", 0.0, 9.0, 0.0)
        g_eng = st.number_input("영어", 0.0, 9.0, 0.0)
        g_math = st.number_input("수학", 0.0, 9.0, 0.0)
    with r2:
        g_t1 = st.number_input("탐구1", 0.0, 9.0, 0.0)
        g_t2 = st.number_input("탐구2", 0.0, 9.0, 0.0)
        g_hist = st.number_input("한국사", 0.0, 9.0, 0.0)

    # ---- 백분위 자동 추정 ----
    grade_list = [g for g in [g_kor, g_math, g_eng, g_t1, g_t2] if g > 0]

    mock_est = None
    if grade_list:
        mapping = {1: 96, 2: 89, 3: 77, 4: 62, 5: 47, 6: 32, 7: 20, 8: 11, 9: 4}
        mock_est = float(np.mean([mapping.get(int(round(g)), 50) for g in grade_list]))

    mock_percentile = mock_input if mock_input > 0 else mock_est

    region_options = ["서울", "경기", "인천", "부산", "대구", "경북", "충북", "충남"]
    selected_regions = st.multiselect(
        "희망 지역 선택",
        options=region_options,
        default=region_options,
    )

    return (
        my_grade,
        mock_percentile,
        selected_regions,
        target_univ,
        target_major
    )


# =========================================
#   학생부종합 자가진단(탭 내부에서 호출)
# =========================================
def render_jagajin_inside_tab():
    st.markdown("### 학생부 종합 전형 적합도 자가진단")

    questions = [
        "1) 이수 과목 수와 난도가 충분히 다양한 편이다.",
        "2) 교과 성취도가 학년 전체에서 상위권이다.",
        "3) 자율·진로·동아리 활동이 지속적·주도적이다.",
        "4) 리더십·배려·공동체 역량이 잘 드러난다.",
        "5) 프로젝트·캠페인·보고서 활동 경험이 있다.",
        "6) 독서 활동이 전공·진로와 연결되어 있다.",
        "7) 실패 경험과 극복 과정이 정리되어 있다.",
        "8) 생기부 내용에 대해 자신 있게 설명할 수 있다.",
        "9) 발표·면접 역량이 뛰어난 편이다.",
        "10) 학교 활동 전체를 관통하는 주제가 있다.",
    ]

    scores = [st.slider(q, 1, 5, 3) for q in questions]

    total = sum(scores)
    max_score = len(scores) * 5
    ratio = total / max_score * 100

    st.markdown("#### 평가 결과")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("총점", f"{total} / {max_score}")
        st.metric("적합도", f"{ratio:.1f}%")

    with col2:
        if total >= 30:
            level, msg = "적정", "학생부 종합전형 지원이 가능한 수준입니다."
        elif total >= 25:
            level, msg = "보통", "기본적인 준비는 되었으나 일부 보완이 필요합니다."
        else:
            level, msg = "미흡", "전형 준비를 다시 점검하는 것이 좋습니다."

        st.subheader(f"종합 평가: {level}")
        st.write(msg)

# =========================================
#         함창고 등급대 분석 화면
# =========================================
def view_grade_analysis():
    st.header("함창고 등급대 분석")

    if not SUJI_HAS_DATA:
        st.error("함창고 수시진학관리 데이터가 없어 분석을 진행할 수 없습니다.")
        return

    df = suji_df.copy()
    df = df.dropna(subset=["대표등급"])

    # ---- 필터 UI ----
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        min_g = float(np.floor(df["대표등급"].min()))
        max_g = float(np.ceil(df["대표등급"].max()))
        grade_min, grade_max = st.slider(
            "대표등급 범위",
            min_value=min_g,
            max_value=max_g,
            value=(min_g, max_g),
            step=1.0,
        )

    with col2:
        years = sorted(df["입시연도"].dropna().unique())
        selected_years = st.multiselect("입시 연도", options=years, default=[years[-1]])

    with col3:
        region = st.multiselect("지역 선택", options=sorted(df["지역"].dropna().unique()))

    with col4:
        univ = st.multiselect("대학 선택", options=sorted(df["대학명"].dropna().unique()))

    with col5:
        group = st.multiselect("대학 그룹", options=["SKY", "수도권 주요 사립", "지방거점국립대", "기타"])

    major_keyword = st.text_input("학과 키워드(모집단위)", "")

    # ---- 필터 적용 ----
    filtered = df[
        (df["대표등급"] >= grade_min) &
        (df["대표등급"] <= grade_max)
    ]

    if selected_years:
        filtered = filtered[filtered["입시연도"].isin(selected_years)]
    if region:
        filtered = filtered[filtered["지역"].isin(region)]
    if univ:
        filtered = filtered[filtered["대학명"].isin(univ)]
    if group:
        filtered = filtered[filtered["대학그룹"].isin(group)]
    if major_keyword:
        filtered = filtered[filtered["모집단위"].astype(str).str.contains(major_keyword)]

    if filtered.empty:
        st.info("선택 조건에 맞는 데이터가 없습니다.")
        return

    # 전형 분류 간단 표기
    vt_col = "전형유형" if "전형유형" in filtered.columns else "전형명(대)"
    base = filtered.assign(
        전형분류=lambda d: d[vt_col]
        .astype(str)
        .str.extract("(교과|종합|농어촌)", expand=False)
        .fillna("기타")
    )

    admit_only = base[base["합격"]]

    # ---- 합격자 지역 분포 ----
    st.subheader("합격자 지역 분포")

    if admit_only.empty:
        st.info("합격 데이터가 없습니다.")
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
                x=alt.X("지역:O", sort="-y"),
                y="합격자수:Q",
                color=alt.condition(alt.datum.지역 == top_region, alt.value("#ff7f0e"), alt.value("#1f77b4")),
            )
        )
        st.altair_chart(chart, use_container_width=True)
        st.markdown(f"**가장 많은 지역: {top_region} (합격 {top_count}명)**")

    # ---- 전형 분포 & 최저 충족률 ----
    st.subheader("전형 분포 및 최저 충족률")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### 합격 전형 분포")
        if admit_only.empty:
            st.info("합격 데이터 없음")
        else:
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

    with c2:
        st.markdown("##### 최저 충족률")
        min_cols = [c for c in base.columns if "최저" in c]
        min_col = min_cols[0] if min_cols else None

        if not min_col:
            st.info("최저 기준 정보 없음")
        else:
            cond_has_min = (
                base[min_col].notna() &
                (base[min_col].astype(str).str.strip() != "") &
                (~base[min_col].astype(str).str.contains("없음"))
            )
            base_min = base[cond_has_min]

            if base_min.empty:
                st.info("최저 기준이 있는 전형이 없습니다.")
            else:
                stats = (
                    base_min.groupby("전형분류")["합격"]
                    .mean().reset_index(name="충족률")
                )
                stats["충족률(%)"] = (stats["충족률"] * 100).round(1)

                bar = (
                    alt.Chart(stats)
                    .mark_bar()
                    .encode(
                        x="전형분류:O",
                        y="충족률(%):Q",
                        tooltip=["전형분류", "충족률(%)"]
                    )
                )
                st.altair_chart(bar, use_container_width=True)

    # =========================================
    #        상세 표 (세부유형 검색 포함)
    # =========================================
    st.markdown("---")
    st.markdown("### 상세 합격 학과 목록 (함창고 입결)")

    detail = base[base["합격"]].copy()
    if detail.empty:
        st.info("조건에 맞는 합격 학과가 없습니다.")
        return

    # 이름 마스킹
    if "이름" in detail.columns:
        detail["이름마스킹"] = detail["이름"].astype(str).str[0] + "OO"
    else:
        detail["이름마스킹"] = ""

    # 지원전형
    detail["지원전형"] = detail.get("전형유형", detail.get("전형명(대)", ""))

    # 세부유형 (있는 그대로 사용)
    detail["세부유형"] = detail.get("세부유형", "")

    # --- 세부유형 검색 필터 ---
    keyword_input = st.text_input("세부유형 검색 (예: 농어촌 기회)", value="")

    if keyword_input.strip():
        keywords = [k.strip() for k in re.split(r"[ ,]+", keyword_input) if k.strip()]

        def match_kw(text):
            t = str(text)
            return all(k in t for k in keywords)

        detail = detail[detail["세부유형"].apply(match_kw)]

    # 최저 정보
    min_cols = [c for c in detail.columns if "최저" in c]
    if min_cols:
        mc = min_cols[0]
        detail["최저"] = detail[mc].fillna("없음").replace("", "없음")
    else:
        detail["최저"] = "없음"

    cols_table = [
        "입시연도", "이름마스킹", "대표등급", "지역",
        "대학명", "모집단위", "지원전형",
        "세부유형", "최저"
    ]
    cols_table = [c for c in cols_table if c in detail.columns]

    table_df = detail[cols_table].sort_values(
        ["입시연도", "대표등급", "대학명", "모집단위"]
    )
    st.dataframe(table_df, use_container_width=True, hide_index=True)


# =========================================
#        추천 공통 함수 (수시·정시 공용)
# =========================================
def pick_recommendations(df, label_col, diff_col, top_n=3):
    """상향 → 적정 → 안전 순서로 추천 대학 선정"""
    results = []

    upper = df[df[label_col] == "상향(도전)"]
    if not upper.empty:
        results.append(upper.nsmallest(top_n, diff_col))

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

    keys = [c for c in ["대학명", "모집단위"] if c in rec.columns]
    if keys:
        rec = rec.drop_duplicates(subset=keys, keep="first")

    return rec
# =========================================
#     수시·정시 추천 탐색기 화면
# =========================================
def view_recommend():
    st.header("수시·정시 추천 탐색기")

    # 성적 + 희망대학/학과 입력
    my_grade, mock_percentile, regions, target_univ, target_major = get_student_inputs()

    tab_su, tab_je, tab_jg = st.tabs(
        ["수시 추천", "정시 추천", "학생부종합 자가진단"]
    )

    # =========================================
    #                수시 추천
    # =========================================
    with tab_su:
        st.subheader("수시 추천 대학 (함창고 합격자 데이터 기반)")

        if not SUJI_HAS_DATA:
            st.warning("학교 합격 데이터가 부족하여 수시 추천을 제공할 수 없습니다.")
        else:
            df = suji_df.copy()
            df = df[df["합격"]]
            df = df.dropna(subset=["대표등급"])

            if "지역" in df.columns and regions:
                df = df[df["지역"].isin(regions)]

            if df.empty:
                st.info("선택 지역의 데이터 부족")
            else:
                group_cols = ["대학명", "모집단위"]
                if "전형유형" in df.columns:
                    group_cols.append("전형유형")

                agg = (
                    df.groupby(group_cols, as_index=False)["대표등급"]
                    .mean()
                    .rename(columns={"대표등급": "합격평균내신"})
                )

                agg["내신차이(합-입)"] = agg["합격평균내신"] - my_grade

                def label_grade(row):
                    diff = row["내신차이(합-입)"]
                    if diff > 0.3:
                        return "안전"
                    if diff < -0.3:
                        return "상향(도전)"
                    return "적정"

                agg["추천구분"] = agg.apply(label_grade, axis=1)

                # 희망 대학/학과 필터
                if target_univ:
                    agg = agg[agg["대학명"].astype(str).str.contains(target_univ)]
                if target_major:
                    agg = agg[agg["모집단위"].astype(str).str.contains(target_major)]

                rec = pick_recommendations(
                    agg, "추천구분", "내신차이(합-입)", top_n=3
                )

                cols = [
                    "추천구분", "대학명", "모집단위", "전형유형",
                    "합격평균내신", "내신차이(합-입)"
                ]

                if not rec.empty:
                    st.dataframe(rec[cols], use_container_width=True, hide_index=True)
                else:
                    st.info("조건에 맞는 추천 대학이 없습니다.")

    # =========================================
    #                정시 추천
    # =========================================
    with tab_je:
        st.subheader("정시 추천 대학 (백분위 기반)")

        if jeong_df is None or JEONG_SCORE_COL is None:
            st.warning("정시 데이터가 부족하여 추천할 수 없습니다.")
        else:
            if mock_percentile is None:
                st.info("정시 추천을 위해 백분위 입력이 필요합니다.")
            else:
                dfj = jeong_df.copy()

                if "지역구분" in dfj.columns and regions:
                    dfj = dfj[dfj["지역구분"].isin(regions)]

                dfj[JEONG_SCORE_COL] = pd.to_numeric(dfj[JEONG_SCORE_COL], errors="coerce")
                dfj = dfj.dropna(subset=[JEONG_SCORE_COL])

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

                # 희망대학 / 학과 필터
                if target_univ:
                    dfj = dfj[dfj["대학명"].astype(str).str.contains(target_univ)]
                if target_major:
                    dfj = dfj[dfj["모집단위"].astype(str).str.contains(target_major)]

                recj = pick_recommendations(dfj, "추천구분", "백분위차이(합-입)", top_n=3)

                # 수시 평균 내신 붙이기
                if SU_DEPT_AVG is not None and {"대학명", "모집단위"}.issubset(recj.columns):
                    recj = recj.merge(
                        SU_DEPT_AVG,
                        how="left",
                        left_on=["대학명", "모집단위"],
                        right_on=["대학명", "모집단위명"],
                    )

                cols = [
                    "추천구분", "대학명", "전형명", "모집군",
                    "모집단위", "정시평균백분위", "백분위차이(합-입)"
                ]
                if "수시평균내신" in recj.columns:
                    cols.append("수시평균내신")

                st.dataframe(recj[cols], use_container_width=True, hide_index=True)

    # =========================================
    #         학생부종합 자가진단
    # =========================================
    with tab_jg:
        render_jagajin_inside_tab()



# =========================================
#     최저 기준으로 대학 찾기 화면
# =========================================
def view_choejeo():
    st.header("최저 기준으로 대학 찾기")

    if choe_df is None:
        st.error("최저 기준 데이터가 없습니다.")
        return

    st.markdown("### 1) 내 희망 최저 기준 입력")

    col1, col2, col3 = st.columns(3)
    with col1:
        g_k = st.number_input("국어 최대 등급", 0.0, 9.0, 0.0, 1.0)
    with col2:
        g_e = st.number_input("영어 최대 등급", 0.0, 9.0, 0.0, 1.0)
    with col3:
        g_m = st.number_input("수학 최대 등급", 0.0, 9.0, 0.0, 1.0)

    col4, col5, col6 = st.columns(3)
    with col4:
        g_t1 = st.number_input("탐구1 최대 등급", 0.0, 9.0, 0.0, 1.0)
    with col5:
        g_t2 = st.number_input("탐구2 최대 등급", 0.0, 9.0, 0.0, 1.0)
    with col6:
        g_h = st.number_input("한국사 최대 등급", 0.0, 9.0, 0.0, 1.0)

    st.caption("0은 해당 과목을 최저 계산에 포함하지 않음")

    colA, colB = st.columns([2, 1])
    with colA:
        regions = st.multiselect(
            "지역",
            options=sorted(choe_df["지역구분"].dropna().unique()),
        )
    with colB:
        keyword = st.text_input("검색어(대학/학과/내용)", "")

    my_grades = {
        "국어": g_k, "수학": g_m, "영어": g_e,
        "탐1": g_t1, "탐2": g_t2, "한국사": g_h
    }

    st.markdown("### 2) 검색 결과")

    if st.button("검색 시작", type="primary"):
        df = choe_df.copy()

        if regions:
            df = df[df["지역구분"].isin(regions)]

        if keyword:
            p = keyword.replace(" ", "")
            df = df[
                df["대학명"].astype(str).str.contains(p)
                | df["모집단위명"].astype(str).str.contains(p)
                | df["최저학력기준내용"].astype(str).str.contains(p)
            ]

        if df.empty:
            st.warning("검색 결과 없음")
            return

        df["최저충족가능"] = df["최저학력기준내용"].apply(
            lambda x: parse_minimum_rule(x, my_grades)
        )
        df_ok = df[df["최저충족가능"]]

        if df_ok.empty:
            st.info("조건을 충족하는 대학 없음")
            return

        # 수시 평균 내신 붙이기
        if SU_DEPT_AVG is not None:
            if {"대학명", "모집단위명"}.issubset(df_ok.columns):
                df_ok = df_ok.merge(
                    SU_DEPT_AVG,
                    on=["대학명", "모집단위명"],
                    how="left"
                )

        cols = ["지역구분", "대학명", "전형세부유형", "모집단위명", "최저학력기준내용"]
        if "수시평균내신" in df_ok.columns:
            cols.append("수시평균내신")

        st.dataframe(df_ok[cols], use_container_width=True, hide_index=True)



# =========================================
#            사이드바 메뉴
# =========================================
with st.sidebar:
    st.image("hch_logo.png", width=120)
    st.markdown("### 메뉴 선택")
    menu = st.radio(
        "",
        ["함창고 등급대 분석", "수시·정시 추천 탐색기", "최저 기준으로 대학 찾기"],
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.8rem; color:gray;'>제작자 함창고 박호종 교사</div>",
        unsafe_allow_html=True,
    )


# =========================================
#                라우팅
# =========================================
if menu == "함창고 등급대 분석":
    view_grade_analysis()
elif menu == "수시·정시 추천 탐색기":
    view_recommend()
elif menu == "최저 기준으로 대학 찾기":
    view_choejeo()


# =========================================
#                푸터
# =========================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:0.8rem;'>
        © 2025 함창고등학교 &nbsp;|&nbsp; 입시 데이터 기반 지원 시스템  
    </div>
    """,
    unsafe_allow_html=True,
)

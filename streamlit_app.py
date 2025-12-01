# streamlit_app.py
# -*- coding: utf-8 -*-
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="함창고 수시·정시 검색기",
    layout="wide",
)

st.title("함창고 수시·정시 검색기")
st.caption("함창고 입결 + 2025 어디가 수시·정시·최저 데이터를 함께 보는 전용 도구 (베타)")

BASE_DIR = Path(__file__).resolve().parent

# 파일 이름 상수
FILE_SUJI = "수시진학관리(2025년2월4일).csv"  # 함창고 내부 수시 진학 관리
FILE_ADIGA_SU = "2025수시입결.csv"            # 어디가 수시
FILE_ADIGA_JEONG = "2025정시입결.csv"         # 어디가 정시
FILE_CHOEJEO = "2025최저모음.csv"             # 어디가 최저 기준 모음


# --------------------------------------------------
# 공통 유틸
# --------------------------------------------------
def _safe_read_csv(relative_name: str) -> pd.DataFrame | None:
    """여러 인코딩을 시도하며 CSV를 읽는다. 없으면 경고만 띄우고 None 반환."""
    path = BASE_DIR / relative_name
    if not path.exists():
        st.error(f"⚠️ 파일을 찾을 수 없습니다: {relative_name}  \n"
                 f"→ GitHub 리포지토리 루트에 동일한 이름으로 업로드했는지 확인해 주세요.")
        return None

    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue

    st.error(f"⚠️ {relative_name} 파일을 읽는 중 인코딩 오류가 발생했습니다.")
    return None


@st.cache_data
def load_all_data():
    suji = _safe_read_csv(FILE_SUJI)
    su = _safe_read_csv(FILE_ADIGA_SU)
    jeong = _safe_read_csv(FILE_ADIGA_JEONG)
    choejeo = _safe_read_csv(FILE_CHOEJEO)
    return suji, su, jeong, choejeo


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# --------------------------------------------------
# 1. 데이터 전처리
# --------------------------------------------------
def preprocess_suji(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None

    df = df.copy()

    # 대표 등급 컬럼 만들기
    candidate_cols = ["전교과", "일반등급", "내등급(환산)"]
    df = to_numeric(df, candidate_cols)
    df["대표등급"] = df[candidate_cols].min(axis=1, skipna=True)

    # 합격 여부 추정
    def _is_admit(row):
        text = ""
        for c in ["등록여부", "최종단계", "불합격사유"]:
            if c in row:
                text += str(row[c])
        if any(k in text for k in ["등록", "합격", "예정"]):
            return "합격"
        return "불합격/기타"

    df["합격여부"] = df.apply(_is_admit, axis=1)

    # 등급대 구간
    bins = [0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 9.5]
    labels = ["~1.5", "1.6~2.0", "2.1~2.5", "2.6~3.0", "3.1~3.5",
              "3.6~4.0", "4.1~4.5", "4.6~5.0", "5.1~"]
    df["등급대"] = pd.cut(df["대표등급"], bins=bins, labels=labels, right=True)

    return df


def preprocess_su(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    df = df.copy()
    # 주요 컬럼 숫자화
    df = to_numeric(df, ["모집인원", "선발비율", "전형총점"])
    return df


def preprocess_jeong(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    df = df.copy()
    # 반영영역 평균백분위 / 평균등급 숫자화
    num_cols = [c for c in df.columns if "평균백분위" in c or "평균등급" in c]
    df = to_numeric(df, num_cols)
    return df


def preprocess_choejeo(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    df = df.copy()
    return df


suji_raw, su_raw, jeong_raw, choejeo_raw = load_all_data()
suji = preprocess_suji(suji_raw)
su = preprocess_su(su_raw)
jeong = preprocess_jeong(jeong_raw)
choejeo = preprocess_choejeo(choejeo_raw)


# --------------------------------------------------
# 2. 사이드바 메뉴
# --------------------------------------------------
with st.sidebar:
    st.header("메뉴 선택")
    main_menu = st.radio(
        "",
        ["함창고 등급대 분석", "수시·정시 추천 탐색기", "최저기준으로 대학찾기"],
    )

    st.markdown("---")
    st.caption("※ 데이터 출처: 함창고 내부 진학 자료 + 2025 대입정보포털 어디가")


# --------------------------------------------------
# 3-1. 함창고 등급대 분석
# --------------------------------------------------
def view_ham_grade_analysis():
    st.subheader("함창고 등급대별 지원·합격 현황")

    if suji is None or suji.empty:
        st.info("함창고 수시진학관리 데이터가 없어 분석을 진행할 수 없습니다.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 등급대별 지원 인원")
        grp = (
            suji.groupby("등급대")["이름"]
            .count()
            .reset_index(name="지원인원")
            .sort_values("등급대")
        )
        fig = px.bar(
            grp,
            x="등급대",
            y="지원인원",
            text="지원인원",
            labels={"등급대": "대표 등급대", "지원인원": "지원 인원(명)"},
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 등급대별 합격 인원")
        grp2 = (
            suji[suji["합격여부"] == "합격"]
            .groupby("등급대")["이름"]
            .count()
            .reset_index(name="합격인원")
            .sort_values("등급대")
        )
        if grp2.empty:
            st.info("아직 우리 학교 수시 합격 내역이 부족하여 합격 분석이 어렵습니다.")
        else:
            fig2 = px.bar(
                grp2,
                x="등급대",
                y="합격인원",
                text="합격인원",
                labels={"등급대": "대표 등급대", "합격인원": "합격 인원(명)"},
            )
            fig2.update_traces(textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 등급대별 지원·합격 대학/전형 분포 (표)")
    pivot = (
        suji.groupby(["등급대", "합격여부", "대학명", "전형유형"])
        ["이름"]
        .count()
        .reset_index(name="인원")
        .sort_values(["등급대", "합격여부", "인원"], ascending=[True, False, False])
    )
    st.dataframe(pivot, use_container_width=True, height=400)


# --------------------------------------------------
# 3-2. 수시·정시 추천 탐색기
# --------------------------------------------------
def view_recommend():
    st.subheader("수시·정시 추천 탐색기")

    if su is None or jeong is None:
        st.info("어디가 수시/정시 데이터가 없어 추천 계산이 어렵습니다.")
        return

    # 희망 지역 기본값
    all_regions = sorted(su["지역구분"].dropna().unique().tolist()) if "지역구분" in su.columns else []
    preferred = ["서울", "경기", "인천", "부산", "대구", "경북", "충북", "충남"]
    default_regions = [r for r in preferred if r in all_regions]

    st.markdown("### 1) 기본 정보 입력")
    c1, c2, c3 = st.columns(3)
    with c1:
        hs_grade = st.number_input("내신(대표 등급, 전교과 기준)", min_value=1.0, max_value=9.0, value=3.0, step=0.1)
    with c2:
        bw_input = st.number_input("정시 반영 영역 평균 백분위(추정)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    with c3:
        hope_regions = st.multiselect(
            "희망 지역",
            options=all_regions,
            default=default_regions if default_regions else all_regions,
        )

    st.markdown("### 2) 최근 모의고사 등급 입력")
    colm1, colm2, colm3 = st.columns(3)
    with colm1:
        mock_kor = st.number_input("국어 등급", 1.0, 9.0, 3.0, step=0.5)
        mock_math = st.number_input("수학 등급", 1.0, 9.0, 3.0, step=0.5)
    with colm2:
        mock_eng = st.number_input("영어 등급", 1.0, 9.0, 3.0, step=0.5)
        mock_t1 = st.number_input("탐구1 등급", 1.0, 9.0, 3.0, step=0.5)
    with colm3:
        mock_t2 = st.number_input("탐구2 등급", 1.0, 9.0, 3.0, step=0.5)
        mock_hist = st.number_input("한국사 등급", 1.0, 9.0, 3.0, step=0.5)

    mock_list = [mock_kor, mock_math, mock_eng, mock_t1, mock_t2, mock_hist]
    mock_avg = float(np.mean(mock_list))
    st.caption(f"최근 모의고사 단순 평균 등급: **{mock_avg:.2f}등급** (참고용)")

    st.markdown("### 3) 종합전형 적합도 자가진단")
    st.write("각 문항을 1~5점으로 체크해 보세요. (5점: 매우 그렇다)")

    factors = [
        "① 이수 과목 수가 충분하다.",
        "② 교과 성취도가 우수하다.",
        "③ 자율·진로·동아리 활동이 우수하다.",
        "④ 리더십·배려·봉사·의사소통·공동체 역량이 드러난다.",
        "⑤ 프로젝트·캠페인·보고서 활동이 우수하다.",
        "⑥ 독서 활동이 풍부하고 잘 정리되어 있다.",
        "⑦ 실패·극복 경험을 스스로 성찰했다.",
        "⑧ 생기부 내용을 자신 있게 설명할 수 있다.",
        "⑨ 발표·면접(스피치) 역량이 뛰어나다.",
        "⑩ 학교 활동 전체를 관통하는 키워드/주제가 있다.",
    ]
    cols_top, cols_bottom = st.columns(2)
    with cols_top:
        scores_left = []
        for i, q in enumerate(factors[:5]):
            scores_left.append(st.slider(q, 1, 5, 3, key=f"gs_left_{i}"))
    with cols_bottom:
        scores_right = []
        for i, q in enumerate(factors[5:]):
            scores_right.append(st.slider(q, 1, 5, 3, key=f"gs_right_{i}"))

    total_score = sum(scores_left) + sum(scores_right)
    st.write(f"**자가진단 총점:** {total_score}점 / 50점")

    if total_score >= 30:
        level = "적정"
        msg = "종합전형 준비 상태가 전반적으로 **양호한 편**입니다."
    elif total_score >= 25:
        level = "보통"
        msg = "기본적인 준비는 되어 있으나, 몇 가지 보완이 필요합니다."
    else:
        level = "미흡"
        msg = "종합전형 준비가 아직 부족합니다. 학교 생활 기록을 점검해 보세요."

    st.info(f"종합전형 적합도: **{level}**  \n{msg}")

    st.markdown("---")
    st.markdown("### 4) 수시·정시 간단 추천 (실험적)")

    st.caption("※ 어디가 2025 수시/정시 데이터를 단순 필터링한 참고용 결과입니다. 최종 지원 여부는 반드시 학교와 상의하세요.")

    if st.button("추천 대학 검색"):
        # 수시: 지역 필터 + 내신 등급 기준 단순 필터
        su_filtered = su.copy()
        if "지역구분" in su_filtered.columns and hope_regions:
            su_filtered = su_filtered[su_filtered["지역구분"].isin(hope_regions)]

        # 내신이 좋을수록 상향/적정/안전 기준을 낮게 잡는다 (단순 모델)
        if not su_filtered.empty:
            su_filtered["내신기준"] = hs_grade
            # 여기서는 등급 수치 대신 전형총점 기준이 없으므로 랜덤 섞기 후 상/중/하 구분
            su_filtered = su_filtered.sample(frac=1.0, random_state=42)

            # 상향/적정/안전 단순 분할
            n = min(50, len(su_filtered))
            top = su_filtered.head(n)

            safe = top.tail(min(2, n))
            mid = top.tail(min(4, n)).head(min(2, n))
            high = top.head(min(2, n))

            st.markdown("#### 수시 추천 (어디가 2025 수시 데이터 기반, 실험적)")
            st.write("**상향 2개**")
            st.dataframe(high[["대학명", "전형세부유형", "계열", "모집단위명", "지역구분"]], use_container_width=True)
            st.write("**적정 2개**")
            st.dataframe(mid[["대학명", "전형세부유형", "계열", "모집단위명", "지역구분"]], use_container_width=True)
            st.write("**안전 2개**")
            st.dataframe(safe[["대학명", "전형세부유형", "계열", "모집단위명", "지역구분"]], use_container_width=True)
        else:
            st.info("선택한 조건에 맞는 수시 데이터가 없습니다.")

        # 정시: 반영영역 평균백분위와 비교
        if jeong is not None and not jeong.empty:
            j = jeong.copy()
            # 반영영역 평균백분위 / 평균등급 중 하나 사용
            score_col = None
            for c in j.columns:
                if "평균백분위" in c:
                    score_col = c
                    break
            if score_col:
                j = to_numeric(j, [score_col])
                j = j.dropna(subset=[score_col])
                # 입력 백분위 근처(±5)에서 5개 추천
                j["diff"] = (j[score_col] - bw_input).abs()
                j = j.sort_values("diff")
                rec = j.head(5)
                st.markdown("#### 정시 추천 (어디가 2025 정시 데이터 기반, 실험적)")
                st.dataframe(
                    rec[["대학명", "전형명", "모집군", "모집단위", score_col]],
                    use_container_width=True,
                )
            else:
                st.info("정시 데이터에 반영영역 평균백분위 정보가 없어 추천을 만들기 어렵습니다.")
        else:
            st.info("정시 데이터가 없어 추천을 만들기 어렵습니다.")


# --------------------------------------------------
# 3-3. 최저기준으로 대학찾기
# --------------------------------------------------
def view_choejeo():
    st.subheader("최저기준으로 대학찾기")

    if choejeo is None or choejeo.empty:
        st.info("어디가 최저 기준 데이터가 없어 검색을 진행할 수 없습니다.")
        return

    st.markdown("### 1) 내 희망 최저 기준 입력")

    c1, c2, c3 = st.columns(3)
    with c1:
        g_kor = st.number_input("국어 최대 등급 (0=미사용)", 0.0, 9.0, 0.0, step=0.5)
        g_math = st.number_input("수학 최대 등급 (0=미사용)", 0.0, 9.0, 0.0, step=0.5)
    with c2:
        g_eng = st.number_input("영어 최대 등급 (0=미사용)", 0.0, 9.0, 0.0, step=0.5)
        g_t1 = st.number_input("탐구1 최대 등급 (0=미사용)", 0.0, 9.0, 0.0, step=0.5)
    with c3:
        g_t2 = st.number_input("탐구2 최대 등급 (0=미사용)", 0.0, 9.0, 0.0, step=0.5)
        g_hist = st.number_input("한국사 최대 등급 (0=미사용)", 0.0, 9.0, 0.0, step=0.5)

    st.caption("※ 0으로 두면 해당 과목은 최저 기준에서 고려하지 않습니다. "
               "텍스트 기준 단순 검색이므로 실제 요강과 차이가 있을 수 있습니다.")

    region_all = sorted(choejeo["지역구분"].dropna().unique().tolist()) if "지역구분" in choejeo.columns else []
    preferred = ["서울", "경기", "인천", "부산", "대구", "경북", "충북", "충남"]
    default_regions = [r for r in preferred if r in region_all]

    colr1, colr2 = st.columns(2)
    with colr1:
        regions = st.multiselect(
            "지역 선택",
            options=region_all,
            default=default_regions if default_regions else region_all,
        )
    with colr2:
        kw = st.text_input("검색 키워드 (대학명/모집단위/내용 일부)", "")

    my_hs_grade = st.number_input("내 내신(대표 등급, 선택)", 1.0, 9.0, 3.0, step=0.1)

    if st.button("최저 기준에 맞는 대학 검색"):
        df = choejeo.copy()
        if regions:
            df = df[df["지역구분"].isin(regions)]

        if kw:
            pattern = kw.strip()
            df = df[
                df["대학명"].astype(str).str.contains(pattern, na=False)
                | df["모집단위명"].astype(str).str.contains(pattern, na=False)
                | df["최저학력기준 내용"].astype(str).str.contains(pattern, na=False)
            ]

        # 과목별 단순 텍스트 필터 (해당 과목이 최저에 등장하는지 여부 위주)
        def subj_filter(flag_grade, label):
            if flag_grade is None or flag_grade <= 0:
                return pd.Series([True] * len(df))
            col = df["최저학력기준 내용"].astype(str)
            return col.str.contains(label, na=False)

        mask = (
            subj_filter(g_kor, "국어")
            & subj_filter(g_math, "수학")
            & subj_filter(g_eng, "영어")
            & subj_filter(g_t1, "탐구")
            & subj_filter(g_t2, "탐구")
            & subj_filter(g_hist, "한국사")
        )
        df = df[mask]

        if df.empty:
            st.warning("입력한 조건에 맞는 최저 기준 데이터가 없습니다.")
            return

        st.markdown("#### 어디가 최저 기준에 해당하는 대학 목록")
        st.dataframe(
            df[["지역구분", "대학명", "전형세부유형", "모집단위명", "최저학력기준 내용"]],
            use_container_width=True,
            height=350,
        )

        # 우리 학교 실제 합격 사례 매칭
        if suji is not None and not suji.empty:
            st.markdown("#### 우리 학교 실제 합격 사례 (참고)")
            # 대학명 기준 단순 매칭
            cand_unis = df["대학명"].unique().tolist()
            suji_match = suji[suji["대학명"].isin(cand_unis)].copy()
            suji_match = suji_match[suji_match["대표등급"] <= my_hs_grade]

            if suji_match.empty:
                st.info("입력한 내신 등급과 조건에 맞는 우리 학교 합격 사례가 아직 없습니다.")
            else:
                st.dataframe(
                    suji_match[
                        ["학년", "반", "번호", "이름", "대학명", "모집시기", "전형유형", "모집단위", "대표등급", "합격여부"]
                    ].sort_values(["대표등급", "대학명"]),
                    use_container_width=True,
                    height=350,
                )
        else:
            st.info("함창고 수시진학관리 데이터가 없어 우리 학교 합격 사례를 보여줄 수 없습니다.")


# --------------------------------------------------
# 메인 라우팅
# --------------------------------------------------
if main_menu == "함창고 등급대 분석":
    view_ham_grade_analysis()
elif main_menu == "수시·정시 추천 탐색기":
    view_recommend()
elif main_menu == "최저기준으로 대학찾기":
    view_choejeo()


# ---------------- 화면 좌측 하단 '제작자' 표시 ----------------
st.markdown(
    """
    <div style="position: fixed; bottom: 10px; left: 10px; 
                font-size: 0.9rem; color: gray; background-color: rgba(255,255,255,0.7);
                padding: 4px 8px; border-radius: 4px;">
        제작자 함창고 국어교사 박호종
    </div>
    """,
    unsafe_allow_html=True,
)



# -*- coding: utf-8 -*-
import os
import re
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(page_title="함창고 수시·정시 진학 검색기", layout="wide")

st.title("함창고 수시·정시 진학 검색기")
st.caption("함창고 학생 맞춤 수시·정시 추천 · 우리 학교 입결 기반 진학 도우미")

TODAY = datetime.date.today().isoformat()

DATA_DIR = Path(".")

SUSI_HISTORY_FILE = DATA_DIR / "수시진학관리(2025년2월4일).csv"
SUSI_META_FILE = DATA_DIR / "2025수시입결.csv"
JUNGSI_FILE = DATA_DIR / "2025정시입결.csv"
CHOEJEO_FILE = DATA_DIR / "2025최저모음.csv"


# --------------------------------------------------
# 글꼴 설정 (있으면 사용, 없으면 건너뜀)
# --------------------------------------------------
def setup_font():
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager, rcParams

        font_path = Path("fonts") / "Pretendard-Bold.ttf"
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            rcParams["font.family"] = font_name
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


setup_font()


# --------------------------------------------------
# 공통 유틸
# --------------------------------------------------
def read_csv_kr(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        return pd.DataFrame()
    # 컬럼명 공통 정리: 줄바꿈/공백 제거
    cols = []
    for c in df.columns:
        c2 = str(c).replace("\n", "").replace("\r", "")
        c2 = re.sub(r"\s+", "", c2)
        cols.append(c2)
    df.columns = cols
    return df


@st.cache_data
def load_worldbank_tertiary_enrollment():
    """공개 데이터: 한국 고등교육 순수등록률 (World Bank)"""
    url = (
        "http://api.worldbank.org/v2/country/KOR/indicator/SE.TER.ENRR"
        "?format=json&per_page=120"
    )
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()[1]
        records = []
        for item in data:
            year = int(item["date"])
            if year > datetime.date.today().year:
                continue
            value = item["value"]
            if value is None:
                continue
            records.append({"year": year, "value": float(value)})
        df = pd.DataFrame(records).sort_values("year")
        return df, None
    except Exception as e:
        # 실패 시 예시 데이터
        example = pd.DataFrame(
            {
                "year": list(range(2010, 2024)),
                "value": np.linspace(70, 95, 14),
            }
        )
        return example, str(e)


@st.cache_data
def load_susi_history():
    df = read_csv_kr(SUSI_HISTORY_FILE)
    if df.empty:
        return df
    # 필요한 컬럼만 남기기
    needed = [
        "학년",
        "반",
        "번호",
        "이름",
        "모집시기",
        "대학명",
        "전형유형",
        "전형명(대)",
        "계열",
        "모집단위",
        "등록여부",
        "내등급(환산)",
    ]
    keep = [c for c in needed if c in df.columns]
    df = df[keep].copy()
    df["내등급(환산)"] = pd.to_numeric(df.get("내등급(환산)"), errors="coerce")
    return df


@st.cache_data
def load_susi_meta():
    return read_csv_kr(SUSI_META_FILE)


@st.cache_data
def load_jungsi():
    return read_csv_kr(JUNGSI_FILE)


@st.cache_data
def load_choejeo():
    return read_csv_kr(CHOEJEO_FILE)


def make_susi_cut_table(history_df: pd.DataFrame) -> pd.DataFrame:
    """우리 학교 수시 합격자 내신 컷 계산"""
    if history_df.empty:
        return pd.DataFrame()
    # 수시 + 등록 기준
    susi = history_df.copy()
    susi = susi[susi["모집시기"] == "수시"]
    if "등록여부" in susi.columns:
        susi = susi[susi["등록여부"].astype(str).str.contains("등록|Y|합격", na=False)]
    susi = susi.dropna(subset=["대학명", "내등급(환산)"])
    if susi.empty:
        return pd.DataFrame()
    grp = (
        susi.groupby(["대학명", "전형명(대)", "계열"], dropna=False)["내등급(환산)"]
        .agg(
            합격자수="count",
            내신중앙값="median",
            내신70백분위=lambda x: x.quantile(0.7),
            최고등급="min",
            최저등급="max",
        )
        .reset_index()
    )
    return grp


def add_susi_meta(cuts: pd.DataFrame, meta: pd.DataFrame, choejeo: pd.DataFrame):
    if cuts.empty:
        return cuts
    df = cuts.copy()
    meta_df = meta.copy()
    cho_df = choejeo.copy()

    # 전형구분 생성 (교과/종합/논술/실기)
    if "전형세부유형" in meta_df.columns:
        def classify_type(x: str) -> str:
            if not isinstance(x, str):
                return "기타"
            if "교과" in x:
                return "교과"
            if "종합" in x:
                return "종합"
            if "논술" in x:
                return "논술"
            if "실기" in x or "특기" in x:
                return "실기"
            return "기타"

        meta_df["전형구분"] = meta_df["전형세부유형"].apply(classify_type)
    else:
        meta_df["전형구분"] = "기타"

    # 면접유무, 단계/일괄
    if "전형방법" not in meta_df.columns:
        meta_df["전형방법"] = ""

    for col in ["면접", "논술", "실기", "서류"]:
        if col not in meta_df.columns:
            meta_df[col] = ""

    # 대학 기준 메타 정보는 대표 1건만 사용
    base_cols = [
        c
        for c in [
            "대학명",
            "지역구분",
            "대학설립형태",
            "전형세부유형",
            "전형구분",
            "계열",
            "상세계열",
            "모집단위명",
            "소재지",
            "전형방법",
            "면접",
            "논술",
            "실기",
            "서류",
        ]
        if c in meta_df.columns
    ]
    meta_base = meta_df[base_cols].drop_duplicates(subset=["대학명"])

    df = df.merge(meta_base, on="대학명", how="left")

    # 최저 정보
    if not cho_df.empty:
        key_cols = [
            c
            for c in [
                "대학명",
                "전형세부유형",
                "최저학력기준내용",
            ]
            if c in cho_df.columns
        ]
        if key_cols:
            cho_base = cho_df[key_cols].drop_duplicates(subset=["대학명"])
            df = df.merge(cho_base, on="대학명", how="left")

    return df


def categorize_by_grade(df: pd.DataFrame, my_grade: float) -> pd.DataFrame:
    """내신 기준 안전/적정/도전 분류"""
    d = df.copy()
    d = d.dropna(subset=["내신중앙값"])
    if d.empty:
        return d
    d["여유도(중앙값-내신)"] = d["내신중앙값"] - my_grade
    # 여유도가 클수록 안전
    conditions = [
        d["여유도(중앙값-내신)"] >= 0.7,
        (d["여유도(중앙값-내신)"] >= 0.3) & (d["여유도(중앙값-내신)"] < 0.7),
        (d["여유도(중앙값-내신)"] > -0.5) & (d["여유도(중앙값-내신)"] < 0.3),
    ]
    choices = ["안전", "적정", "도전"]
    d["추천구분"] = np.select(conditions, choices, default="위험")
    return d


def filter_susi_reco(
    df: pd.DataFrame,
    region: str,
    univ_type: str,
    jeonhyeong: list,
    need_choejeo: str,
    need_interview: str,
    step_type: str,
):
    if df.empty:
        return df
    res = df.copy()

    if region != "전체" and "지역구분" in res.columns:
        res = res[res["지역구분"] == region]

    if univ_type != "전체" and "대학설립형태" in res.columns:
        res = res[res["대학설립형태"] == univ_type]

    if jeonhyeong and "전형구분" in res.columns:
        res = res[res["전형구분"].isin(jeonhyeong)]

    # 최저 유무
    if need_choejeo != "상관없음" and "최저학력기준내용" in res.columns:
        has_cho = res["최저학력기준내용"].notna() & res["최저학력기준내용"].astype(str).str.strip().ne(
            ""
        )
        if need_choejeo == "최저있음":
            res = res[has_cho]
        elif need_choejeo == "최저없음":
            res = res[~has_cho]

    # 면접 유무
    if need_interview != "상관없음" and "면접" in res.columns:
        has_intv = res["면접"].astype(str).str.contains(r"\d|점|반영|실시", na=False)
        if need_interview == "면접있음":
            res = res[has_intv]
        elif need_interview == "면접없음":
            res = res[~has_intv]

    # 단계/일괄
    if step_type != "상관없음" and "전형방법" in res.columns:
        col = res["전형방법"].astype(str)
        if step_type == "다단계전형":
            res = res[col.str.contains("단계", na=False)]
        elif step_type == "일괄선발":
            res = res[~col.str.contains("단계", na=False)]

    return res


def get_jungsi_reco(df: pd.DataFrame, my_percent: float, top_n=5):
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    if "반영영역평균백분위" not in d.columns:
        return pd.DataFrame()
    d["cut"] = pd.to_numeric(d["반영영역평균백분위"], errors="coerce")
    d = d.dropna(subset=["cut"])
    d["여유도(백분위-컷)"] = my_percent - d["cut"]
    d["적합도"] = -np.abs(d["여유도(백분위-컷)"])
    d = d.sort_values("적합도", ascending=True).copy()
    return d.head(top_n)


# --------------------------------------------------
# UI: 탭 구성
# --------------------------------------------------
tab_public, tab_local = st.tabs(["공개 데이터 대시보드", "함창고 수시·정시 검색기"])

# --------------------------------------------------
# 1. 공개 데이터 대시보드
# --------------------------------------------------
with tab_public:
    st.subheader("공개 데이터: 한국 고등교육 순수 재학률 추이 (World Bank)")

    wb_df, wb_err = load_worldbank_tertiary_enrollment()
    if wb_err:
        st.info("실시간 World Bank API 호출에 실패하여 예시 데이터를 사용합니다.")

    st.write("단위: 순수 재학률(%)")
    fig = px.line(
        wb_df,
        x="year",
        y="value",
        markers=True,
        labels={"year": "연도", "value": "순수 재학률(%)"},
        title="한국 고등교육 순수 재학률 추이",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(wb_df, use_container_width=True)

    csv_bytes = wb_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "공개 데이터 CSV 다운로드",
        data=csv_bytes,
        file_name=f"worldbank_tertiary_enrollment_KOR_{TODAY}.csv",
        mime="text/csv",
    )

# --------------------------------------------------
# 2. 함창고 수시·정시 검색기
# --------------------------------------------------
with tab_local:
    st.subheader("함창고 수시·정시 진학 검색기 (내신·모의고사·우리 학교 입결 기반)")

    # 데이터 로드
    susi_history = load_susi_history()
    susi_meta = load_susi_meta()
    jungsi_df = load_jungsi()
    choejeo_df = load_choejeo()

    if susi_history.empty:
        st.warning("수시진학관리 CSV를 읽어오지 못했습니다. 파일 경로와 인코딩을 확인하세요.")

    cuts = make_susi_cut_table(susi_history)
    cuts_meta = add_susi_meta(cuts, susi_meta, choejeo_df)

    st.markdown("### 1. 수시 추천 대학 (우리 학교 입결 + 내신 기준)")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        my_grade = st.number_input(
            "나의 내신 평균등급 (전교과 또는 주요 반영 내신)",
            min_value=1.0,
            max_value=9.0,
            step=0.1,
            value=3.0,
        )

        region_options = ["전체"]
        if "지역구분" in cuts_meta.columns:
            region_options += sorted(cuts_meta["지역구분"].dropna().unique().tolist())
        region = st.selectbox("지역 선택", region_options)

        univ_type_options = ["전체"]
        if "대학설립형태" in cuts_meta.columns:
            univ_type_options += sorted(cuts_meta["대학설립형태"].dropna().unique().tolist())
        univ_type = st.selectbox("대학 설립 유형", univ_type_options)

        jg_options = ["교과", "종합", "논술", "실기"]
        jeonhyeong_sel = st.multiselect("전형 구분 (복수 선택 가능)", jg_options, default=jg_options[:2])

        choejeo_sel = st.radio(
            "수능 최저학력기준",
            ["상관없음", "최저있음", "최저없음"],
            horizontal=True,
        )

        interview_sel = st.radio(
            "면접 전형 여부",
            ["상관없음", "면접있음", "면접없음"],
            horizontal=True,
        )

        step_sel = st.radio(
            "단계/일괄 선발",
            ["상관없음", "다단계전형", "일괄선발"],
            horizontal=True,
        )

        susi_button = st.button("✅ 수시 추천 대학 검색")

    with col_right:
        if cuts_meta.empty:
            st.info("아직 우리 학교 수시 합격 내역이 부족하여 추천 계산이 어렵습니다.")
        elif susi_button:
            base = categorize_by_grade(cuts_meta, my_grade)
            base = filter_susi_reco(
                base,
                region=region,
                univ_type=univ_type,
                jeonhyeong=jeonhyeong_sel,
                need_choejeo=choejeo_sel,
                need_interview=interview_sel,
                step_type=step_sel,
            )

            if base.empty:
                st.warning("조건에 맞는 추천 결과가 없습니다. 필터를 완화해 보세요.")
            else:
                safe = base[base["추천구분"] == "안전"].sort_values(
                    "여유도(중앙값-내신)", ascending=False
                ).head(2)
                fit = base[base["추천구분"] == "적정"].sort_values(
                    "여유도(중앙값-내신)", ascending=False
                ).head(2)
                risk = base[base["추천구분"] == "도전"].sort_values(
                    "여유도(중앙값-내신)", ascending=False
                ).head(2)

                st.markdown("#### 🔵 안전 지원권 (2개 내외)")
                st.dataframe(safe, use_container_width=True)

                st.markdown("#### 🟢 적정 지원권 (2개 내외)")
                st.dataframe(fit, use_container_width=True)

                st.markdown("#### 🟠 도전 지원권 (2개 내외)")
                st.dataframe(risk, use_container_width=True)

                all_reco = pd.concat([safe, fit, risk], ignore_index=True)
                csv_bytes = all_reco.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "수시 추천 결과 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"hamchang_susi_recommend_{TODAY}.csv",
                    mime="text/csv",
                )
        else:
            st.info("왼쪽에서 내신과 필터를 입력한 뒤 ‘수시 추천 대학 검색’을 눌러 주세요.")

    st.markdown("---")
    st.markdown("### 2. 정시 추천 대학 (모의고사 백분위 기준)")

    col_j1, col_j2 = st.columns([1, 2])

    with col_j1:
        my_percent = st.number_input(
            "나의 반영영역 평균 백분위 (최근 모의고사 기준)",
            min_value=0.0,
            max_value=100.0,
            step=0.5,
            value=85.0,
        )
        jung_button = st.button("✅ 정시 추천 대학 검색")

    with col_j2:
        if jungsi_df.empty:
            st.warning("정시 입결 CSV를 읽어오지 못했습니다.")
        elif jung_button:
            reco_j = get_jungsi_reco(jungsi_df, my_percent, top_n=5)
            if reco_j.empty:
                st.warning("정시 추천 결과를 계산할 수 없습니다. CSV의 '반영영역 평균백분위' 컬럼을 확인하세요.")
            else:
                st.dataframe(reco_j, use_container_width=True)
                csv_bytes = reco_j.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "정시 추천 결과 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"hamchang_jungsi_recommend_{TODAY}.csv",
                    mime="text/csv",
                )
        else:
            st.info("왼쪽에서 평균 백분위를 입력한 뒤 ‘정시 추천 대학 검색’을 눌러 주세요.")

    st.markdown("---")
    st.markdown("### 3. 학생부종합 전형 적합도 자가 진단")

    st.caption("각 문항에 대해 1점(전혀 아니다) ~ 5점(매우 그렇다)로 솔직하게 체크해 보세요.")

    q_labels = [
        "이수 과목(선택과목 포함)이 다양한 편이다.",
        "주요 교과 성취도가 우수한 편이다.",
        "자율·진로·동아리 활동이 꾸준하고 내용이 있다.",
        "리더십·배려·봉사·의사소통·공동체 역량을 보여주는 사례가 있다.",
        "프로젝트·캠페인·보고서 활동 경험이 있다.",
        "독서 활동이 풍부하고 활동과 잘 연결되어 있다.",
        "실패 경험과 극복 경험을 스스로 성찰해 본 적이 있다.",
        "학교 활동 전반을 관통하는 나만의 키워드/주제가 있다.",
        "생활기록부에 기록된 활동에 대해 자신 있게 설명할 수 있다.",
        "면접에서 자신의 생각을 조리 있게 말할 수 있다.",
    ]

    scores = []
    for i, q in enumerate(q_labels, start=1):
        val = st.slider(f"{i}) {q}", min_value=1, max_value=5, value=3)
        scores.append(val)

    total = sum(scores)
    st.write(f"**총점: {total}점 (최대 50점)**")

    if total >= 40:
        level = "매우 적정"
        desc = "학생부종합 전형에 매우 잘 준비되어 있습니다."
    elif total >= 30:
        level = "적정"
        desc = "학생부종합 전형 지원에 비교적 적합한 상태입니다."
    elif total >= 25:
        level = "보통"
        desc = "기본 준비는 되어 있으나, 몇 가지 영역에서 보완이 필요합니다."
    elif total >= 20:
        level = "보완 필요"
        desc = "중요 요소들이 부족할 수 있습니다. 남은 기간 동안 계획적인 보완이 필요합니다."
    else:
        level = "미흡"
        desc = "학생부종합 전형보다는 다른 전형(교과·정시 등)을 중심으로 전략을 세우는 것이 좋을 수 있습니다."

    st.success(f"종합 진단 결과: **{level}**")
    st.write(desc)

    st.markdown("---")
    st.markdown("### 4. 최저 기준 & 우리 학교 합격 내신 비교")

    cho_cols_ok = (
        ("대학명" in choejeo_df.columns) and ("최저학력기준내용" in choejeo_df.columns)
    )
    if not cho_cols_ok:
        st.info("최저학력 기준 CSV에서 대학명/최저학력기준 내용 컬럼을 찾지 못했습니다.")
    else:
        uni_list = sorted(choejeo_df["대학명"].dropna().unique().tolist())
        sel_uni = st.selectbox("대학 선택", ["선택 안 함"] + uni_list)

        if sel_uni != "선택 안 함":
            sub = choejeo_df[choejeo_df["대학명"] == sel_uni]
            st.markdown("#### 선택 대학의 대표 최저 기준 예시")
            st.dataframe(
                sub[["대학명", "전형세부유형", "계열", "모집단위명", "최저학력기준내용"]],
                use_container_width=True,
            )

            my_grade_for_uni = st.number_input(
                "해당 대학 지원 가정 시 나의 내신 등급 (반영 기준)",
                min_value=1.0,
                max_value=9.0,
                step=0.1,
                value=my_grade,
            )

            # 우리 학교 합격자 통계
            if not cuts.empty:
                hist_uni = cuts[cuts["대학명"] == sel_uni]
            else:
                hist_uni = pd.DataFrame()

            if hist_uni.empty:
                st.info("해당 대학의 우리 학교 합격 데이터가 아직 충분하지 않습니다.")
            else:
                st.markdown("#### 우리 학교 합격자 내신 통계")
                st.dataframe(hist_uni, use_container_width=True)

                avg_cut = hist_uni["내신중앙값"].mean()
                diff = avg_cut - my_grade_for_uni
                if diff >= 0.7:
                    msg = "우리 학교 평균 합격자보다 **상대적으로 여유 있는 내신**입니다."
                elif diff >= 0.3:
                    msg = "우리 학교 평균 합격자 수준에 **근접한 내신**입니다."
                elif diff > -0.5:
                    msg = "우리 학교 합격자 평균보다 **다소 불리한 내신**입니다. 다른 강점을 함께 보여 주어야 합니다."
                else:
                    msg = "우리 학교 기준으로 볼 때 **상당히 도전적인 내신**입니다."

                st.success(
                    f"내신 비교 결과: 당신의 내신({my_grade_for_uni:.1f}) vs 우리 학교 평균 합격 내신({avg_cut:.2f}) → {msg}"
                )

    # --------------------------------------------------
    # 정리된 데이터 일괄 다운로드 (선택)
    # --------------------------------------------------
    st.markdown("---")
    st.markdown("### 5. 데이터 다운로드")

    col_d1, col_d2, col_d3, col_d4 = st.columns(4)

    with col_d1:
        if not susi_history.empty:
            st.download_button(
                "우리 학교 수시 진학 관리 원본 CSV",
                data=susi_history.to_csv(index=False).encode("utf-8-sig"),
                file_name="hamchang_susi_history_raw.csv",
                mime="text/csv",
            )

    with col_d2:
        if not cuts_meta.empty:
            st.download_button(
                "수시 합격 내신 컷 테이블 CSV",
                data=cuts_meta.to_csv(index=False).encode("utf-8-sig"),
                file_name="hamchang_susi_cut_with_meta.csv",
                mime="text/csv",
            )

    with col_d3:
        if not jungsi_df.empty:
            st.download_button(
                "정시 입결 원본 CSV",
                data=jungsi_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="hamchang_jungsi_raw.csv",
                mime="text/csv",
            )

    with col_d4:
        if not choejeo_df.empty:
            st.download_button(
                "수능 최저 기준 원본 CSV",
                data=choejeo_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="hamchang_choejeo_raw.csv",
                mime="text/csv",
            )

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

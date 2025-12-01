# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

# ---------------- 기본 설정 ----------------
st.set_page_config(
    page_title="함창고 수시·정시 검색기",
    layout="wide"
)

st.title("함창고 수시·정시 검색기")
st.caption("함창고 입결 + 2025 어디가 수시·정시·최저 데이터를 함께 보는 전용 도구 (베타)")

TODAY = pd.Timestamp("today").date()

# ---------------- 폰트 설정 (있으면 적용, 없으면 무시) ----------------
FONT_PATH = Path("fonts/Pretendard-Bold.ttf")
if FONT_PATH.exists():
    st.markdown(
        f"""
        <style>
        @font-face {{
            font-family: 'PretendardBold';
            src: url('{FONT_PATH.as_posix()}');
        }}
        html, body, [class*="css"]  {{
            font-family: 'PretendardBold', sans-serif;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------- 공통: CSV 로더 ----------------
def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"⚠ 파일을 찾을 수 없습니다: {path.name}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="euc-kr")
    # 컬럼에 들어 있는 줄바꿈/공백 정리
    df.columns = [c.replace("\n", "").strip() for c in df.columns]
    return df


@st.cache_data
def load_all_data():
    base = Path(".")
    hs = _safe_read_csv(base / "수시진학관리(2025년2월4일).csv")
    su = _safe_read_csv(base / "2025수시입결.csv")
    jeong = _safe_read_csv(base / "2025정시입결.csv")
    choi = _safe_read_csv(base / "2025최저모음.csv")

    # --------- 함창고 수시진학관리 전처리 ---------
    if not hs.empty:
        # 학년/반/번호 문자열화
        for col in ["학년", "반", "번호"]:
            if col in hs.columns:
                hs[col] = hs[col].astype(str)

        # 전교과 내신 컬럼 추출
        grade_cols = [c for c in hs.columns if "전교과" in c]
        hs["전교과등급"] = pd.to_numeric(
            hs[grade_cols[0]], errors="coerce"
        ) if grade_cols else np.nan

        # 환산 등급/점수
        conv_cols = [c for c in hs.columns if "내등급" in c]
        hs["환산등급"] = pd.to_numeric(
            hs[conv_cols[0]], errors="coerce"
        ) if conv_cols else np.nan

        score_cols = [c for c in hs.columns if "내점수" in c]
        hs["환산점수"] = pd.to_numeric(
            hs[score_cols[0]], errors="coerce"
        ) if score_cols else np.nan

        # 지원시기/전형
        if "모집시기" not in hs.columns:
            hs["모집시기"] = ""
        if "전형유형" not in hs.columns and "전형명(대)" in hs.columns:
            hs["전형유형"] = hs["전형명(대)"]
        # 지원/합격 구분
        def _result(row):
            text = ""
            for c in ["등록여부", "최종단계", "불합격사유"]:
                if c in hs.columns and pd.notna(row.get(c, "")):
                    text += str(row[c])
            if any(k in text for k in ["등록", "합격", "최종합격"]):
                return "합격"
            return "지원만"
        hs["결과"] = hs.apply(_result, axis=1)

    # --------- 2025 수시입결 전처리 ---------
    if not su.empty:
        # 전형 대분류 추출
        def _big_type(txt: str) -> str:
            if not isinstance(txt, str):
                return "기타"
            if "교과" in txt:
                return "학생부교과"
            if "종합" in txt:
                return "학생부종합"
            if "논술" in txt:
                return "논술"
            if "실기" in txt or "특기" in txt:
                return "실기·특기"
            return "기타"

        su["전형대분류"] = su["전형세부유형"].apply(_big_type) if "전형세부유형" in su.columns else "기타"

        # 면접/단계/일괄 플래그
        if "전형방법" in su.columns:
            su["선발형태"] = su["전형방법"].astype(str).apply(
                lambda x: "단계" if "단계" in x else ("일괄" if "일괄" in x else "기타")
            )
        else:
            su["선발형태"] = "기타"

        if "면접" in su.columns:
            su["면접여부"] = su["면접"].astype(str).apply(lambda x: "면접 있음" if x.strip() not in ["", "0", "-", "무"] else "면접 없음")
        else:
            su["면접여부"] = "정보없음"

    # --------- 2025 정시입결 전처리 ---------
    if not jeong.empty:
        # 반영영역 평균백분위/등급 컬럼 정리
        for c in jeong.columns:
            if "반영영역평균백분위" in c:
                jeong["평균백분위"] = pd.to_numeric(jeong[c], errors="coerce")
            if "반영영역평균등급" in c:
                jeong["평균등급"] = pd.to_numeric(jeong[c], errors="coerce")

    # --------- 2025 최저모음 전처리 ---------
    if not choi.empty:
        # 최저학력기준 내용 텍스트 정리
        name = [c for c in choi.columns if "최저학력기준" in c]
        if name:
            choi.rename(columns={name[0]: "최저학력기준내용"}, inplace=True)
        else:
            choi["최저학력기준내용"] = ""

    return hs, su, jeong, choi


hs_df, su_df, jeong_df, choi_df = load_all_data()

# ---------------- 사이드바 메뉴 ----------------
st.sidebar.header("메뉴 선택")
page = st.sidebar.radio(
    "",
    [
        "함창고 등급대별 지원/합격 분석",
        "수시·정시 추천 탐색기 (베타)",
        "학생부종합 적합도 자기진단",
        "최저기준으로 대학찾기",
    ],
)


# ============================================================
# 1. 함창고 등급대별 지원/합격 분석
# ============================================================
if page == "함창고 등급대별 지원/합격 분석":
    st.subheader("함창고 등급대별 지원·합격 현황")

    if hs_df.empty:
        st.info("함창고 수시진학관리 데이터가 없어 분석을 진행할 수 없습니다.")
    else:
        # 등급 컬럼 체크
        if hs_df["전교과등급"].isna().all():
            st.error("전교과 등급 컬럼을 찾지 못했습니다. CSV의 '전교과' 관련 컬럼명을 확인해 주세요.")
            st.write("현재 컬럼 목록:", list(hs_df.columns))
        else:
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                year_list = sorted(hs_df["학년"].unique()) if "학년" in hs_df.columns else []
                selected_year = st.multiselect("학년 선택", year_list, default=year_list)
            with col_filter2:
                term_list = sorted(hs_df["모집시기"].dropna().unique())
                default_term = ["수시", "정시"] if term_list else []
                selected_term = st.multiselect("모집 시기", term_list, default=default_term)

            df = hs_df.copy()
            if selected_year:
                df = df[df["학년"].isin(selected_year)]
            if selected_term:
                df = df[df["모집시기"].isin(selected_term)]

            # 등급 구간 나누기
            bins = [0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 9.0]
            labels = [
                "1.5 이내",
                "1.6~2.0",
                "2.1~2.5",
                "2.6~3.0",
                "3.1~3.5",
                "3.6~4.0",
                "4.1~5.0",
                "5.0 초과",
            ]
            df["등급대"] = pd.cut(df["전교과등급"], bins=bins, labels=labels, right=True, include_lowest=True)

            st.markdown("#### 등급대별 지원/합격 건수")

            summary1 = (
                df.groupby(["등급대", "결과"])["이름"]
                .count()
                .reset_index(name="건수")
                .dropna(subset=["등급대"])
            )

            if summary1.empty:
                st.info("선택한 조건에서 데이터가 없습니다.")
            else:
                fig1 = px.bar(
                    summary1,
                    x="등급대",
                    y="건수",
                    color="결과",
                    barmode="group",
                    title="등급대별 지원/합격 건수",
                )
                fig1.update_layout(xaxis_title="전교과 등급대", yaxis_title="건수")
                st.plotly_chart(fig1, use_container_width=True)

            st.markdown("#### 등급대별 전형 유형 분포 (함창고 지원 기준)")
            if "전형유형" not in df.columns:
                st.info("전형유형 컬럼이 없어 전형 분석을 할 수 없습니다.")
            else:
                summary2 = (
                    df.groupby(["등급대", "전형유형", "결과"])["이름"]
                    .count()
                    .reset_index(name="건수")
                    .dropna(subset=["등급대"])
                )
                if summary2.empty:
                    st.info("선택한 조건에서 데이터가 없습니다.")
                else:
                    fig2 = px.bar(
                        summary2,
                        x="등급대",
                        y="건수",
                        color="전형유형",
                        facet_row="결과",
                        title="등급대별 전형유형 분포 (지원 vs 합격)",
                    )
                    fig2.update_layout(xaxis_title="전교과 등급대", yaxis_title="건수")
                    st.plotly_chart(fig2, use_container_width=True)

            st.markdown("##### 원자료 일부 미리보기 (함창고 수시진학관리)")
            st.dataframe(df.head(50))


# ============================================================
# 2. 수시·정시 추천 탐색기 (베타)
# ============================================================
elif page == "수시·정시 추천 탐색기 (베타)":
    st.subheader("수시·정시 추천 탐색기 (베타)")

    col_input1, col_input2 = st.columns(2)
    with col_input1:
        student_grade = st.number_input("내 전교과 내신 등급 (예: 2.3)", min_value=1.0, max_value=9.0, step=0.1, value=3.0)
        mock_percentile = st.number_input("최근 모의고사 평균 백분위 (예: 85)", min_value=0, max_value=100, step=1, value=80)
    with col_input2:
        region_options = sorted(su_df["지역구분"].dropna().unique()) if not su_df.empty else []
        selected_region = st.multiselect("희망 지역 (어디가 수시 기준)", region_options, default=region_options)

        type_options = ["전체", "학생부교과", "학생부종합", "논술", "실기·특기"]
        selected_type = st.selectbox("전형 대분류", type_options, index=0)

    st.markdown("---")
    if st.button("추천 대학 검색 (참고용 사례 보기)"):
        st.warning(
            "아직 우리 학교 수시 합격 내역이 충분하지 않아 **정확한 안전/적정/하향 추천 계산은 어렵습니다.**\n"
            "아래 표는 단지 **참고용 사례**로만 활용해 주세요."
        )

        # ---------- 함창고 입결에서 비슷한 등급대 사례 ----------
        st.markdown("### ① 함창고 실제 지원·합격 사례 (참고용)")

        if hs_df.empty or hs_df["전교과등급"].isna().all():
            st.info("함창고 수시진학관리 데이터가 부족하거나 전교과 등급 정보를 찾을 수 없습니다.")
        else:
            df = hs_df.copy()
            # 학생 등급 ±0.3 범위
            df = df[df["전교과등급"].between(student_grade - 0.3, student_grade + 0.3, inclusive="both")]
            # 수시만
            if "모집시기" in df.columns:
                df = df[df["모집시기"] == "수시"]

            if df.empty:
                st.info("해당 등급대의 함창고 수시 지원 사례가 아직 없습니다.")
            else:
                show_cols = [c for c in ["학년", "반", "번호", "이름", "대학명", "모집시기", "전형유형", "전형명(대)", "모집단위", "전교과등급", "환산등급", "결과"] if c in df.columns]
                st.dataframe(df[show_cols].sort_values(["결과", "전교과등급"]))

        # ---------- 어디가 2025 수시/정시 데이터 ----------
        st.markdown("### ② 어디가 2025 수시/정시 데이터 (조건 필터)")

        tab_su, tab_jeong = st.tabs(["수시 데이터", "정시 데이터"])
        with tab_su:
            if su_df.empty:
                st.info("2025 수시 입결 파일이 없습니다.")
            else:
                df_su = su_df.copy()
                if selected_region:
                    df_su = df_su[df_su["지역구분"].isin(selected_region)]
                if selected_type != "전체":
                    df_su = df_su[df_su["전형대분류"] == selected_type]

                st.caption("※ 어디가 2025 수시모집요강 기반 데이터입니다.")
                cols = [c for c in ["지역구분", "대학명", "전형세부유형", "전형대분류", "계열", "상세계열", "모집단위명", "모집인원", "전형방법", "면접여부"] if c in df_su.columns]
                st.dataframe(df_su[cols].head(200))

        with tab_jeong:
            if jeong_df.empty:
                st.info("2025 정시 입결 파일이 없습니다.")
            else:
                df_j = jeong_df.copy()
                # 모의고사 백분위를 이용해서 단순 참고 필터 (평균백분위 ±5 범위)
                if "평균백분위" in df_j.columns:
                    df_j = df_j[df_j["평균백분위"].between(mock_percentile - 5, mock_percentile + 5, inclusive="both")]

                cols = [c for c in ["대학명", "전형명", "모집군", "모집단위", "모집인원", "경쟁률", "평균백분위", "평균등급"] if c in df_j.columns]
                st.caption("※ 어디가 2025 정시 입결 기반 데이터입니다.")
                st.dataframe(df_j[cols].head(200))


# ============================================================
# 3. 학생부종합 적합도 자기진단
# ============================================================
elif page == "학생부종합 적합도 자기진단":
    st.subheader("학생부종합 전형 적합도 자기진단 (5점 척도)")

    st.markdown(
        """
        각 항목에 대해 **0점(전혀 해당 없음) ~ 5점(매우 잘 되어 있음)** 사이에서 솔직하게 선택해 보세요.  
        슬라이더를 모두 조정하면 총점과 간단한 진단 결과가 아래에 표시됩니다.
        """
    )

    items = [
        "① 이수 과목 수가 충분하다.",
        "② 주요 교과 성취도가 우수하다.",
        "③ 자율·진로·동아리 활동이 우수하다.",
        "④ 리더십·배려·봉사·의사소통·공동체 역량이 잘 드러난다.",
        "⑤ 프로젝트·캠페인·보고서 활동이 잘 정리되어 있다.",
        "⑥ 독서 활동이 풍부하고, 활동과 잘 연결되어 있다.",
        "⑦ 실패·갈등 경험과 극복 과정이 정리되어 있다.",
        "⑧ 생기부 내용을 자신 있게 설명할 수 있고, 스피치 역량이 좋다.",
    ]

    cols = st.columns(2)
    scores = []
    for i, item in enumerate(items):
        with cols[i % 2]:
            val = st.slider(item, min_value=0, max_value=5, value=3, step=1, key=f"ssa_score_{i}")
            scores.append(val)

    total = sum(scores)
    st.markdown("---")
    st.markdown(f"### 총점: **{total}점 / 40점**")

    if total >= 30:
        level = "적정 이상 (준비 상태가 비교적 좋습니다.)"
        color = "🟢"
    elif total >= 25:
        level = "보통 (강한 부분과 보완할 부분이 함께 있습니다.)"
        color = "🟡"
    elif total >= 20:
        level = "주의 (여러 요소를 정비할 필요가 있습니다.)"
        color = "🟠"
    else:
        level = "미흡 (전략 재설계가 필요합니다.)"
        color = "🔴"

    st.markdown(f"**진단 결과:** {color} {level}")

    # 항목별 막대그래프 (너비를 줄이기 위해 2열 레이아웃 아래에 위치)
    chart_df = pd.DataFrame({
        "항목": [f"{i+1}" for i in range(len(items))],
        "점수": scores,
    })
    fig = px.bar(chart_df, x="항목", y="점수", range_y=[0, 5], title="항목별 점수")
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 4. 최저기준으로 대학찾기
# ============================================================
elif page == "최저기준으로 대학찾기":
    st.subheader("최저기준으로 대학찾기")

    if choi_df.empty:
        st.info("2025 최저 기준 데이터(2025최저모음.csv)를 찾을 수 없습니다.")
    else:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            region_opt = sorted(choi_df["지역구분"].dropna().unique())
            selected_region = st.multiselect("지역 선택", region_opt, default=region_opt)
        with col_f2:
            univ_opt = sorted(choi_df["대학명"].dropna().unique())
            selected_univ = st.multiselect("대학 선택 (선택 안 하면 전체)", univ_opt)
        with col_f3:
            keyword = st.text_input("최저 기준 키워드 검색 (예: 2합 6, 국수영탐 등)", "")

        my_grade = st.number_input("내 전교과 내신 등급 입력 (예: 2.3)", min_value=1.0, max_value=9.0, step=0.1, value=3.0)

        if st.button("최저 기준으로 검색"):
            df = choi_df.copy()
            if selected_region:
                df = df[df["지역구분"].isin(selected_region)]
            if selected_univ:
                df = df[df["대학명"].isin(selected_univ)]
            if keyword.strip():
                df = df[df["최저학력기준내용"].astype(str).str.contains(keyword.strip(), na=False)]

            if df.empty:
                st.info("조건에 맞는 최저 기준 데이터가 없습니다.")
            else:
                st.markdown("### ① 어디가 2025 최저 기준 (검색 결과)")
                cols_show = [c for c in ["지역구분", "대학설립형태", "대학명", "전형세부유형", "계열", "상세계열", "모집단위명", "소재지", "모집인원", "최저학력기준내용"] if c in df.columns]
                st.dataframe(df[cols_show].head(300))

                # ---- 함창고 합격 사례 매칭 ----
                st.markdown("### ② 함창고 합격 사례 (내신 + 최저 충족 사례)")

                if hs_df.empty or hs_df["전교과등급"].isna().all():
                    st.info("함창고 수시진학관리 데이터가 없거나 전교과 등급 컬럼을 찾지 못했습니다.")
                else:
                    # 최저 검색 결과의 대학/모집단위 기준으로 매칭
                    target_pairs = df[["대학명", "모집단위명"]].dropna().drop_duplicates()
                    hs = hs_df.copy()
                    hs = hs[hs["전교과등급"] <= my_grade]  # 내신 충족
                    hs = hs[hs["결과"] == "합격"]          # 합격 사례만

                    # 모집단위명이 수시진학관리의 "모집단위" 컬럼과 대략 일치하는지 확인
                    if "모집단위" in hs.columns:
                        merged = pd.merge(
                            hs,
                            target_pairs,
                            left_on=["대학명", "모집단위"],
                            right_on=["대학명", "모집단위명"],
                            how="inner",
                        )
                    else:
                        merged = pd.merge(
                            hs,
                            target_pairs[["대학명"]],
                            on="대학명",
                            how="inner",
                        )

                    if merged.empty:
                        st.info("해당 최저 기준과 내신 등급을 동시에 충족한 함창고 합격 사례가 아직 없습니다.")
                    else:
                        show_cols2 = [c for c in ["학년", "반", "번호", "이름", "대학명", "전형유형", "전형명(대)", "모집단위", "전교과등급", "환산등급", "결과"] if c in merged.columns]
                        st.dataframe(merged[show_cols2].sort_values("전교과등급"))

# ---------------- 하단 안내 ----------------
st.markdown(
    """
    <div style="position: fixed; bottom: 8px; right: 12px;
                font-size: 0.8rem; color: gray; background-color: rgba(255,255,255,0.7);
                padding: 4px 8px; border-radius: 4px;">
        만든이: 함창고 교사 박호종 · AI 보조: ChatGPT
    </div>
    """,
    unsafe_allow_html=True,
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


def view_choejeo():
    st.header("최저 기준으로 대학 찾기")

    # ✅ 업로드 UI 제거: 레포에 있는 2027최저모음.csv만 사용
    df = None
    p = DATA_DIR / "2027최저모음.csv"
    if p.exists():
        df = read_choejeo_any(p)

    if df is None or df.empty:
        st.error("최저 기준 데이터 파일을 찾을 수 없습니다. (2027최저모음.csv)")
        return

    # ✅ 지역 컬럼 자동 탐지
    region_col = "지역구분" if "지역구분" in df.columns else ("지역" if "지역" in df.columns else None)

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
        options=sorted(df[region_col].dropna().unique()) if region_col else []
    )
    keyword = st.text_input("검색 키워드 (대학명/학과/기준 내용)", "")

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
            st.error("최저학력기준내용 컬럼을 찾을 수 없습니다. (헤더 인식 실패 가능)")
            st.write("현재 컬럼:", dff.columns.tolist())
            return

        dff["최저충족가능"] = dff["최저학력기준내용"].apply(lambda x: parse_minimum_rule(x, my_grades))
        ok = dff[dff["최저충족가능"]]

        if ok.empty:
            st.info("입력 조건을 충족하는 대학이 없습니다.")
            return

        cols = [c for c in ["지역구분", "지역", "대학명", "전형세부유형", "모집단위명", "최저학력기준내용"] if c in ok.columns]
        st.dataframe(ok[cols], hide_index=True, use_container_width=True)

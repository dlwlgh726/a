# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="금리와 아파트 가격 예측", layout="wide")
st.title("🏡 금리와 아파트 매매가격의 관계 분석 (2006~2024)")

# 파일 업로드
apt_file = st.file_uploader("📁 아파트 매매 실거래 평균가격 CSV 업로드", type="csv")
rate_file = st.file_uploader("📁 한국은행 금리 CSV 업로드", type="csv")

# 파일 로딩 함수 (인코딩 자동)
def load_csv(file):
    try:
        return pd.read_csv(file, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(file, encoding="cp949")

if apt_file and rate_file:
    try:
        # CSV 불러오기
        apt_df = load_csv(apt_file)
        rate_df = load_csv(rate_file)

        # 날짜 컬럼 정리
        apt_df = apt_df.rename(columns={apt_df.columns[0]: "날짜"})
        rate_df = rate_df.rename(columns={rate_df.columns[0]: "날짜"})

        apt_df["날짜"] = pd.to_datetime(apt_df["날짜"], errors="coerce")
        rate_df["날짜"] = pd.to_datetime(rate_df["날짜"], errors="coerce")

        apt_df = apt_df.dropna(subset=["날짜"])
        rate_df = rate_df.dropna(subset=["날짜"])

        # 연도 추출
        apt_df["연도"] = apt_df["날짜"].dt.year
        rate_df["연도"] = rate_df["날짜"].dt.year

        # 연도 필터링: 2006~2024년
        apt_df = apt_df[(apt_df["연도"] >= 2006) & (apt_df["연도"] <= 2024)]
        rate_df = rate_df[(rate_df["연도"] >= 2006) & (rate_df["연도"] <= 2024)]

        # 연도별 평균 계산
        apt_year = apt_df.groupby("연도").mean(numeric_only=True).reset_index()
        rate_year = rate_df.groupby("연도").mean(numeric_only=True).reset_index()

        # 병합
        merged = pd.merge(apt_year, rate_year, on="연도", how="inner")

        # 주요 컬럼 자동 탐색
        price_col = [col for col in merged.columns if "가격" in col][0]
        rate_col = [col for col in merged.columns if "금리" in col][0]

        # NaN 제거
        merged = merged.dropna(subset=[price_col, rate_col])

        # 데이터 부족 예외 처리
        if merged.shape[0] < 2:
            st.error("🚫 유효한 데이터가 2행 미만입니다. 분석을 진행할 수 없습니다.")
            st.dataframe(merged)
            st.stop()

        # 📊 시각화
        st.subheader("📈 연도별 금리 vs 아파트 평균가격")
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(merged["연도"], merged[price_col], color='tab:blue', marker='o', label='아파트 가격')
        ax1.set_ylabel("아파트 가격", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(merged["연도"], merged[rate_col], color='tab:red', marker='s', label='기준금리')
        ax2.set_ylabel("기준금리", color="tab:red")
        st.pyplot(fig)

        # 선형 회귀
        st.subheader("🔍 선형 회귀 분석")
        X = merged[[rate_col]].dropna()
        y = merged[price_col].loc[X.index]  # X와 인덱스 일치

        if len(X) < 2:
            st.error("🚫 학습 가능한 데이터가 부족합니다. 최소 2개 이상 필요합니다.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.markdown(f"**📊 R² Score:** `{r2_score(y_test, y_pred):.4f}`")
        st.markdown(f"**📈 회귀 계수 (기울기):** `{model.coef_[0]:,.2f}`")
        st.markdown(f"**📉 절편:** `{model.intercept_:,.2f}`")

        # 회귀선 시각화
        fig2, ax = plt.subplots()
        sns.regplot(x=rate_col, y=price_col, data=merged, ax=ax, ci=None, line_kws={"color": "red"})
        ax.set_xlabel("기준금리")
        ax.set_ylabel("아파트 가격")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
else:
    st.info("⏳ 두 개의 CSV 파일을 업로드해 주세요.")

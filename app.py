# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------------
# 1. 데이터 로딩 및 전처리
# ------------------------

@st.cache_data
def load_and_train():
    # 데이터 로딩
    apt_df = pd.read_csv("아파트_매매_실거래_평균가격_20250609090955.csv", encoding="cp949")
    rate_df = pd.read_csv("한국은행 기준금리 및 여수신금리_05123930.csv", encoding="cp949")

    # melt
    apt_long = apt_df.melt(id_vars=["행정구역별"], var_name="연도", value_name="평균가격")
    apt_long = apt_long[apt_long["행정구역별"] == "전국"].drop(columns=["행정구역별"])
    apt_long["연도"] = apt_long["연도"].astype(int)
    apt_long["평균가격"] = pd.to_numeric(apt_long["평균가격"], errors="coerce")

    rate_long = rate_df[rate_df["계정항목"] == "한국은행 기준금리"].drop(columns=["계정항목"])
    rate_long = rate_long.melt(var_name="연도", value_name="기준금리")
    rate_long["연도"] = rate_long["연도"].astype(int)
    rate_long["기준금리"] = pd.to_numeric(rate_long["기준금리"], errors="coerce")

    # 병합
    merged = pd.merge(apt_long, rate_long, on="연도", how="inner")

    # 모델 학습
    X = merged[["기준금리"]]
    y = merged["평균가격"]
    model = LinearRegression()
    model.fit(X, y)

    return model, merged

model, data = load_and_train()

# ------------------------
# 2. 웹앱 UI
# ------------------------

st.set_page_config(page_title="금리 기반 아파트 가격 예측기", layout="centered")

st.title("🏡 금리 기반 아파트 평균가격 예측기")
st.write("한국은행 기준금리를 입력하면, 예상 전국 평균 아파트 매매가격을 예측합니다.")

# 사용자 입력
input_rate = st.slider("📉 기준금리 (%)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)

# 예측
predicted_price = model.predict(np.array([[input_rate]]))[0]

# 결과 출력
st.metric("📊 예상 평균 아파트 가격", f"{predicted_price:,.0f} 백만원")

# 차트 표시
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()
sns.regplot(x="기준금리", y="평균가격", data=data, ax=ax)
ax.scatter(input_rate, predicted_price, color="red", label="입력값")
ax.set_title("기준금리와 평균 아파트 가격")
ax.legend()
st.pyplot(fig)

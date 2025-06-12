import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="지역별 금리 기반 아파트 가격 예측기", layout="centered")
st.title("🏠 지역별 금리 기반 아파트 평균가격 예측기")

# ------------------------
# 1. 데이터 로딩
# ------------------------

@st.cache_data
def load_data():
    # 지역별 아파트 가격
    apt_df = pd.read_csv("아파트_매매_실거래_평균가격_20250611110831.csv", encoding="cp949")
    apt_df = apt_df.rename(columns={"행정구역별(2)": "지역"})
    apt_long = apt_df.melt(id_vars=["지역"], var_name="연도", value_name="평균가격")
    apt_long["연도"] = apt_long["연도"].astype(int)
    apt_long["평균가격"] = pd.to_numeric(apt_long["평균가격"], errors="coerce")

    # 금리 데이터
    rate_df = pd.read_csv("한국은행 기준금리 및 여수신금리_05123930.csv", encoding="cp949")
    rate_df = rate_df[rate_df["계정항목"] == "한국은행 기준금리"].drop(columns=["계정항목"])
    rate_long = rate_df.melt(var_name="연도", value_name="기준금리")
    rate_long["연도"] = rate_long["연도"].astype(int)
    rate_long["기준금리"] = pd.to_numeric(rate_long["기준금리"], errors="coerce")

    # 병합
    merged = pd.merge(apt_long, rate_long, on="연도", how="inner")
    return merged

data = load_data()

# ------------------------
# 2. 지역 선택 및 모델 학습
# ------------------------
regions = sorted(data["지역"].unique())
selected_region = st.selectbox("📍 지역을 선택하세요", regions)
input_rate = st.slider("📉 기준금리 (%)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)

region_data = data[data["지역"] == selected_region].dropna()

# 선형회귀 모델 학습
X = region_data[["기준금리"]]
y = region_data["평균가격"]
model = LinearRegression()
model.fit(X, y)
predicted_price = model.predict(np.array([[input_rate]]))[0]

# ------------------------
# 3. 상관계수 계산 및 출력
# ------------------------
corr = region_data["기준금리"].corr(region_data["평균가격"])

st.subheader(f"🔍 {selected_region} 지역 기준금리 {input_rate:.1f}%에 대한 예측")
st.metric("📊 예상 평균 아파트 가격", f"{predicted_price:,.0f} 백만원")
st.write(f"📈 기준금리와 아파트 평균가격 간 상관계수: **{corr:.3f}**")

# ------------------------
# 4. 기준금리와 평균가격 산점도 및 회귀선 시각화
# ------------------------
fig, ax = plt.subplots()
sns.regplot(x="기준금리", y="평균가격", data=region_data, ax=ax, scatter_kws={"s": 50})
ax.scatter(input_rate, predicted_price, color="red", label="입력값", s=100)
ax.set_title(f"[ {selected_region} ] 기준금리와 아파트 평균가격 관계")
ax.set_xlabel("기준금리 (%)")
ax.set_ylabel("평균 아파트 가격 (백만원)")
ax.legend()
st.pyplot(fig)

# ------------------------
# 5. 연도별 변화 추이 그래프
# ------------------------
fig2, ax1 = plt.subplots(figsize=(8, 4))

color1 = "tab:blue"
ax1.set_xlabel("연도")
ax1.set_ylabel("평균 아파트 가격 (백만원)", color=color1)
ax1.plot(region_data["연도"], region_data["평균가격"], marker='o', color=color1, label="평균가격")
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = "tab:red"
ax2.set_ylabel("기준금리 (%)", color=color2)
ax2.plot(region_data["연도"], region_data["기준금리"], marker='s', linestyle='--', color=color2, label="기준금리")
ax2.tick_params(axis='y', labelcolor=color2)

plt.title(f"[ {selected_region} ] 연도별 평균 아파트 가격 및 기준금리 변화 추이")
fig2.tight_layout()
st.pyplot(fig2)

import streamlit as st

# ✅ Streamlit 페이지 설정
st.set_page_config(page_title="지역별 금리 기반 아파트 가격 예측기", layout="centered")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.font_manager as fm
import os

# ------------------------
# 0. 한글 폰트 설정 (NanumGothic-Regular.ttf 사용)
# ------------------------
def set_korean_font():
    font_path = "NanumGothic-Regular.ttf"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['axes.unicode_minus'] = False
    else:
        print("❗ NanumGothic-Regular.ttf 파일을 찾을 수 없습니다.")

set_korean_font()

# ------------------------
# 1. 제목
# ------------------------
st.title("\U0001F3E0 지역별 금리 기반 아파트 평균가격 예측기")

# ------------------------
# 2. 데이터 로딩
# ------------------------
@st.cache_data
def load_data():
    # 아파트 가격 데이터
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

    return pd.merge(apt_long, rate_long, on="연도", how="inner")

data = load_data()

# ------------------------
# 3. 사용자 입력
# ------------------------
regions = sorted(data["지역"].unique())
selected_region = st.selectbox("\U0001F4CD 지역을 선택하세요", regions)
input_rate = st.slider("\U0001F4C9 기준금리 (%)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)

region_data = data[data["지역"] == selected_region].dropna()

# ------------------------
# 4. 모델 학습 및 예측 (가중치 적용)
# ------------------------
X = region_data[["기준금리"]]
y = region_data["평균가격"]

# 최근 연도에 더 높은 가중치 부여
weights = np.exp((region_data["연도"] - region_data["연도"].min()) / 2)
model = LinearRegression()
model.fit(X, y, sample_weight=weights)

predicted_price = model.predict(np.array([[input_rate]]))[0]

# ------------------------
# 5. 결과 출력
# ------------------------
corr = region_data["기준금리"].corr(region_data["평균가격"])

st.subheader(f"\U0001F50D {selected_region} 지역 기준금리 {input_rate:.1f}%에 대한 예측")
st.metric("\U0001F4CA 예상 평균 아파트 가격", f"{predicted_price:,.0f} 백만원")
st.write(f"\U0001F4C8 기준금리와 아파트 평균가격 간 상관계수: **{corr:.3f}**")

# ------------------------
# 6. 산점도 + 회귀선 그래프
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
# 7. 연도별 아파트 가격 및 금리 변화 추이
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

# ------------------------

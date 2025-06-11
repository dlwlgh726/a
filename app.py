import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ----------- 1. 데이터 불러오기 및 전처리 -----------

# CSV 파일 경로 (필요 시 변경)
apt_path = "아파트_매매_실거래_평균가격_20250609090955.csv"
rate_path = "한국은행 기준금리 및 여수신금리_05123930.csv"

# 인코딩 주의 (한글 CSV는 일반적으로 cp949)
apt_df = pd.read_csv(apt_path, encoding="cp949")
rate_df = pd.read_csv(rate_path, encoding="cp949")

# 아파트 가격 데이터: '전국'만 추출 + melt
apt_long = apt_df.melt(id_vars=["행정구역별"], var_name="연도", value_name="평균가격")
apt_long = apt_long[apt_long["행정구역별"] == "전국"].drop(columns=["행정구역별"])
apt_long["연도"] = apt_long["연도"].astype(int)
apt_long["평균가격"] = pd.to_numeric(apt_long["평균가격"], errors="coerce")

# 금리 데이터: 기준금리만 추출 + melt
rate_long = rate_df[rate_df["계정항목"] == "한국은행 기준금리"].drop(columns=["계정항목"])
rate_long = rate_long.melt(var_name="연도", value_name="기준금리")
rate_long["연도"] = rate_long["연도"].astype(int)
rate_long["기준금리"] = pd.to_numeric(rate_long["기준금리"], errors="coerce")

# 병합
merged_df = pd.merge(apt_long, rate_long, on="연도", how="inner")

# ----------- 2. 상관관계 분석 -----------

print("\n📊 상관계수:")
print(merged_df.corr(numeric_only=True))

sns.heatmap(merged_df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("상관관계 히트맵")
plt.show()

# ----------- 3. 머신러닝: 선형 회귀 -----------

# 입력 변수(X), 타겟 변수(y)
X = merged_df[["기준금리"]]
y = merged_df["평균가격"]

# 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# ----------- 4. 성능 평가 -----------

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📈 MAE (평균절대오차): {mae:.2f}")
print(f"📈 R² score: {r2:.4f}")

# 시각화
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test["기준금리"], y=y_test, label="실제값")
sns.lineplot(x=X_test["기준금리"], y=y_pred, color="red", label="예측값")
plt.title("기준금리로 예측한 평균 아파트 가격")
plt.xlabel("기준금리 (%)")
plt.ylabel("평균 아파트 가격")
plt.legend()
plt.show()

# ----------- 5. 예측 공식 출력 -----------

coef = model.coef_[0]
intercept = model.intercept_
print(f"\n📌 예측 공식: 평균 아파트 가격 = {coef:.2f} * 기준금리 + {intercept:.2f}")

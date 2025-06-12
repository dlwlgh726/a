import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "NanumGothic-Regular.ttf"
fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()
plt.rcParams['axes.unicode_minus'] = False

plt.figure()
plt.title("테스트: 기준금리와 아파트 평균가격 관계")
plt.xlabel("기준금리 (%)")
plt.ylabel("평균 아파트 가격 (백만원)")
plt.plot([1, 2, 3], [200, 150, 180])
plt.show()

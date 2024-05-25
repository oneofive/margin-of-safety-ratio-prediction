import pandas as pd # type: ignore
import tensorflow as tf
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore

# 데이터 로드
data = pd.read_csv('hospital.csv')

# 문자열을 숫자로 변환
data['medical_profits'] = data['medical_profits'].str.replace(',', '').astype(int)
data['medical_expenses'] = data['medical_expenses'].str.replace(',', '').astype(int)

# 안전한계율 계산
data['safety_margin_rate'] = (data['medical_profits'] - data['medical_expenses']) / data['medical_profits']

# 독립 변수와 종속 변수 분리
X = data['area_population'].values.reshape(-1, 1)
y = data['safety_margin_rate'].values.reshape(-1, 1)

# 데이터 정규화
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 모델 정의
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
model.fit(X_train, y_train, epochs=100)

# 모델 평가
loss = model.evaluate(X_test, y_test)
print('Test Loss:', loss)

# 새로운 인구수 예측
new_population = np.array([[450000]])
new_population_scaled = scaler_X.transform(new_population)

predicted_safety_margin_rate_scaled = model.predict(new_population_scaled)
predicted_safety_margin_rate = scaler_y.inverse_transform(predicted_safety_margin_rate_scaled)
print('Predicted Safety Margin Rate:', predicted_safety_margin_rate)

# 테스트 데이터에 대한 예측
y_test_pred_scaled = model.predict(X_test)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

# 실제 값과 예측 값 비교
y_test_actual = scaler_y.inverse_transform(y_test)

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test_actual, color='blue', label='Actual Safety Margin Rate')
plt.scatter(X_test, y_test_pred, color='red', label='Predicted Safety Margin Rate')
plt.title('Actual vs Predicted Safety Margin Rate')
plt.xlabel('Area Population')
plt.ylabel('Safety Margin Rate')
plt.legend()
plt.show()
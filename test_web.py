import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Cấu hình giao diện trang web
st.set_page_config(
    page_title='Dự đoán Tiểu đường', page_icon='🩺', layout='centered'
)

# Tiêu đề ứng dụng
st.title('🩺 Ứng dụng Dự đoán Nguy cơ Tiểu đường')
st.write(
    'Nhập các thông số sức khỏe bên dưới để hệ thống phân tích và dự đoán nguy'
    ' cơ mắc bệnh.'
)


# Load mô hình và scaler (Sử dụng cache để tối ưu tốc độ)
@st.cache_resource
def load_models():
  model = joblib.load('diabetes_model.pkl')
  scaler = joblib.load('scaler.pkl')
  return model, scaler


try:
  model, scaler = load_models()
except Exception as e:
  st.error(
      f'Không tìm thấy file mô hình (.pkl). Hãy đảm bảo `diabetes_model.pkl` và'
      f' `scaler.pkl` nằm chung thư mục với file app.py! Lỗi: {e}'
  )
  st.stop()

# --- KHU VỰC NHẬP DỮ LIỆU TỪ NGƯỜI DÙNG ---
st.subheader('📝 Nhập thông số sức khỏe')

# Chia thành 2 cột cho giao diện cân đối, dễ nhìn
col1, col2 = st.columns(2)

with col1:
  age = st.number_input(
      'Tuổi (age)', min_value=0.0, max_value=120.0, value=45.0, step=1.0
  )
  bmi = st.number_input(
      'Chỉ số BMI (bmi)', min_value=10.0, max_value=70.0, value=25.0
  )
  HbA1c_level = st.number_input(
      'Mức HbA1c (HbA1c_level)', min_value=3.0, max_value=15.0, value=5.5
  )

with col2:
  blood_glucose_level = st.number_input(
      'Đường huyết (blood_glucose_level)',
      min_value=50.0,
      max_value=300.0,
      value=100.0,
  )

  # Dùng selectbox cho hypertension (0 hoặc 1)
  hypertension_choice = st.selectbox(
      'Tăng huyết áp (hypertension)',
      options=[0, 1],
      format_func=lambda x: (
          '0 - Không' if x == 0 else '1 - Có tiền sử tăng huyết áp'
      ),
  )

  # Dùng selectbox cho heart_disease (0 hoặc 1)
  heart_disease_choice = st.selectbox(
      'Bệnh tim (heart_disease)',
      options=[0, 1],
      format_func=lambda x: '0 - Không' if x == 0 else '1 - Có tiền sử bệnh tim',
  )

# Nút bấm thực hiện dự đoán
if st.button('🔍 Dự đoán ngay', type='primary'):
  # 1. Đưa dữ liệu đầu vào thành DataFrame (Đảm bảo thứ tự cột KHỚP HOÀN TOÀN lúc train)
  input_data = pd.DataFrame(
      {
          'age': [age],
          'hypertension': [hypertension_choice],
          'heart_disease': [heart_disease_choice],
          'bmi': [bmi],
          'HbA1c_level': [HbA1c_level],
          'blood_glucose_level': [blood_glucose_level],
      }
  )

  # 2. Chuẩn hóa dữ liệu bằng scaler đã load
  # (Lưu ý: StandardScaler thường chỉ chuẩn hóa các cột số thực như age, bmi, HbA1c, glucose;
  # tuy nhiên nếu lúc train bạn đã gom chung tất cả 6 cột vào scaler thì code này chạy hoàn toàn bình thường)
  input_scaled = scaler.transform(input_data)

  # 3. Dự đoán bằng mô hình
  prediction = model.predict(input_scaled)[0]
  probability = model.predict_proba(input_scaled)[0]

  # 4. Hiển thị kết quả ra màn hình web
  st.divider()
  st.subheader('📊 Kết quả phân tích:')

  if prediction == 1:
    st.error(
        f'⚠️ **Cảnh báo:** Mô hình dự đoán có nguy cơ **MẮC BỆNH TIỂU ĐƯỜNG**!'
        f' (Xác suất: {probability[1]*100:.2f}%)'
    )
  else:
    st.success(
        f'✅ **An toàn:** Mô hình dự đoán **KHÔNG CÓ DẤU HIỆU** mắc bệnh.'
        f' (Xác suất an toàn: {probability[0]*100:.2f}%)'
    )

import joblib
import pandas as pd

# 1. Tải mô hình, scaler, tập dữ liệu x_test và y_test (true labels) từ các file đã lưu
print("Đang tải dữ liệu và mô hình...")
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

X_test = pd.read_csv('x_test.csv')
# Giả sử bạn có lưu nhãn thực tế của tập test vào file y_test.csv
y_test = pd.read_csv('y_test.csv') 

# 2. Chuẩn hóa dữ liệu x_test bằng scaler đã load
X_test_scaled = scaler.transform(X_test)

# 3. Thực hiện dự đoán hàng loạt cho toàn bộ tập test
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)

# --- XỬ LÝ DỮ LIỆU KẾT QUẢ ---
# Tạo một DataFrame chứa kết quả để dễ lọc và hiển thị
results_df = X_test.copy()
results_df['True_Label'] = y_test.values.ravel() # Gắn nhãn thực tế vào
results_df['Predicted_Label'] = predictions
results_df['Max_Probability'] = [max(prob) for prob in probabilities]

# Đếm tổng số lượng dự đoán là 1
count_ones = (predictions == 1).sum()
total_samples = len(predictions)
print(f"\n-> Tổng số ca được dự đoán là 1 (Có bệnh): {count_ones} / {total_samples} mẫu")

# --- LỌC VÀ IN 5 HÀNG ĐẦU DỰ ĐOÁN LÀ 1 VÀ 0 (CÓ TRUE LABEL) ---
print("\n" + "="*50)
print("5 HÀNG ĐẦU TIÊN CÓ DỰ ĐOÁN LÀ 1 (Có bệnh):")
print("="*50)
df_pred_1 = results_df[results_df['Predicted_Label'] == 1]
print(df_pred_1.head(5))

print("\n" + "="*50)
print("5 HÀNG ĐẦU TIÊN CÓ DỰ ĐOÁN LÀ 0 (Không bệnh):")
print("="*50)
df_pred_0 = results_df[results_df['Predicted_Label'] == 0]
print(df_pred_0.head(5))

# 4. Xuất toàn bộ kết quả ra file CSV mới
results_df.to_csv('x_test_predicted_results.csv', index=False)
print(
    '\nĐã dự đoán xong và lưu toàn bộ kết quả vào file'
    " 'x_test_predicted_results.csv' thành công!"
)

import joblib

# Load scaler đã lưu từ trước
scaler = joblib.load("scaler.pkl")

print("mean =", [float(x) for x in scaler.mean_])
print("scale =", [float(x) for x in scaler.scale_])

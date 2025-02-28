import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. ĐỌC EXCEL
# Đọc tệp Excel
file_path = "DoanhThu_QuanCaPhe.xlsx"
df = pd.read_excel(file_path)
print("Dữ liệu gốc:")
print(df.head())

# 2. LÀM SẠCH DỮ LIỆU
# Chuyển đổi kiểu dữ liệu
df["Ngày"] = pd.to_datetime(df["Ngày"], errors='coerce')

# Xử lý lỗi: In ra các giá trị thiếu, lỗi, sai định dạng
print("\nCác giá trị thiếu:")
print(df.isnull().sum())

# Loại bỏ các dòng có giá trị thiếu
df_cleaned = df.dropna()

# In lại dữ liệu sau khi xử lý
print("\nDữ liệu sau khi làm sạch:")
print(df_cleaned.head())

# 3. EDA
# 3.1. Tóm tắt dữ liệu
print("\nThông tin dữ liệu:")
df_cleaned.info()
print("\nThống kê mô tả:")
print(df_cleaned.describe())

# 3.2. Thống kê
print("\nCác chỉ số thống kê:")
print("Trung bình:")
print(df_cleaned.mean(numeric_only=True))
print("\nTrung vị:")
print(df_cleaned.median(numeric_only=True))
print("\nPhương sai:")
print(df_cleaned.var(numeric_only=True))
print("\nGiá trị nhỏ nhất:")
print(df_cleaned.min(numeric_only=True))
print("\nGiá trị lớn nhất:")
print(df_cleaned.max(numeric_only=True))

# 3.3. Kiểm tra giá trị thiếu
missing_values = df_cleaned.isnull().sum()
print("\nGiá trị thiếu sau khi làm sạch:")
print(missing_values)

# 3.4. Trực quan hóa dữ liệu
plt.figure(figsize=(10, 5))
sns.lineplot(x=df_cleaned["Ngày"], y=df_cleaned["Doanh thu (VND)"], marker='o')
plt.xticks(rotation=45)
plt.xlabel("Ngày")
plt.ylabel("Doanh thu (VND)")
plt.title("Xu hướng doanh thu theo ngày")
plt.show()

# Biểu đồ so sánh doanh thu giữa các nhân viên
plt.figure(figsize=(10, 5))
sns.barplot(x=df_cleaned["Người bán"], y=df_cleaned["Doanh thu (VND)"], palette="viridis")
plt.xticks(rotation=45)
plt.xlabel("Nhân viên")
plt.ylabel("Doanh thu (VND)")
plt.title("Doanh thu của từng nhân viên")
plt.show()

# Biểu đồ sản phẩm bán chạy nhất
top_selling = df_cleaned["Sản phẩm bán chạy"].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_selling.index, y=top_selling.values, palette="coolwarm")
plt.xticks(rotation=45)
plt.xlabel("Sản phẩm")
plt.ylabel("Số lần bán")
plt.title("Top 10 sản phẩm bán chạy nhất")
plt.show()

# 4. Mô hình hồi quy tuyến tính
X = df_cleaned[["Số lượng bán"]]
y = df_cleaned["Doanh thu (VND)"]

# 4.1. Chia tập dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nĐánh giá mô hình hồi quy tuyến tính:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# Vẽ biểu đồ hồi quy
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_test["Số lượng bán"], y=y_test, color="blue", label="Thực tế")
sns.lineplot(x=X_test["Số lượng bán"], y=y_pred, color="red", label="Dự đoán")
plt.xlabel("Số lượng bán")
plt.ylabel("Doanh thu (VND)")
plt.title("Mô hình hồi quy tuyến tính: Số lượng bán vs Doanh thu")
plt.legend()
plt.show()


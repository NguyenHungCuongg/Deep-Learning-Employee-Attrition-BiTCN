Giai đoạn thực nghiệm (Experiments) trong bài báo được thực hiện một cách toàn diện để chứng minh tính hiệu quả của mô hình **Bi-TCN** đề xuất thông qua việc so sánh với nhiều mô hình cơ sở và đánh giá trên hai bộ dữ liệu khác nhau.

Dưới đây là tóm tắt chi tiết các thông tin kỹ thuật và kết quả thực nghiệm:

### 1. Thiết lập thực nghiệm và Bộ dữ liệu

Bài báo sử dụng hai bộ dữ liệu công khai để đánh giá:

- **IBM HR Analytics:** Gồm **1.470 bản ghi** và **34 đặc trưng**; nhãn mục tiêu là "Attrition" (lớp thiểu số chiếm khoảng 16,12%).
- **Kaggle Employee Churn:** Gồm **14.249 mẫu** và **10 đặc trưng** (sau khi làm sạch còn 12.717 mẫu).
- **Chiến lược kiểm định:** Sử dụng **5-fold cross-validation** để đảm bảo tính ổn định và tránh phụ thuộc vào một cách chia tập duy nhất.
- **Chỉ số đánh giá:** Accuracy, Precision, Recall, F1-score, AUC và thời gian huấn luyện.

### 2. Thiết lập siêu tham số và Kiến trúc (Table 4)

Tất cả các mô hình học sâu (DL) trong bài báo đều được cấu hình thống nhất để đảm bảo so sánh công bằng:

- **Tham số huấn luyện:** 50 Epochs, trình tối ưu hóa **Adam (lr=0.001)**, hàm mất mát Binary Crossentropy, **Batch Size 128**, và tỷ lệ Dropout 0.5.
- **Kiến trúc Bi-TCN đề xuất:**
  - **Lớp 1:** 32 filters, kích thước kernel = 3.
  - **Lớp 2:** 64 filters, kích thước kernel = 5.
  - **Dilation:** Cả hai lớp đều sử dụng hệ số giãn $$ để mở rộng trường thụ cảm.
  - **Fully Connected:** 128 đơn vị.

### 3. Kết quả trên bộ dữ liệu IBM (Table 5 & 6)

- **So sánh với Baselines:** Mô hình Bi-TCN đạt **Accuracy 89,65%** và **F1-score 61,61%**, vượt qua các mô hình mạnh như Transformer (88,77%) và Bi-GRU (86,87%).
- **Tác động của Data Augmentation:**
  - Bài báo sinh thêm **3.000 mẫu dữ liệu nhân tạo** (1.500 mẫu cho mỗi lớp) bằng GAN để cân bằng dữ liệu.
  - Khi kết hợp **Bi-TCN + GAN**, độ chính xác tăng lên **92,17%** và AUC đạt **84,95%**.
  - So sánh với các phương pháp khác (SMOTE, ADASYN), GAN cho độ chính xác cao nhất dù ADASYN đạt F1-score tốt hơn một chút.

### 4. Kết quả trên bộ dữ liệu Kaggle (Table 7)

Mô hình Bi-TCN tiếp tục thể hiện hiệu năng vượt trội trên tập dữ liệu quy mô lớn hơn:

- **Accuracy:** Đạt **97,83%** (chỉ thấp hơn Random Forest 0,02% nhưng cao hơn tất cả các mô hình DL khác).
- **F1-score:** Đạt **95,56%** (cao nhất trong tất cả các mô hình, bao gồm cả Random Forest).
- **AUC:** Đạt **96,94%**, khẳng định khả năng phân biệt lớp cực kỳ chính xác.

### 5. Giải thích mô hình (Explainable AI - SHAP)

Bài báo sử dụng kỹ thuật SHAP để xác định các đặc trưng quan trọng nhất dẫn đến quyết định nghỉ việc:

- **IBM Dataset:** Các yếu tố hàng đầu là **StockOptionLevel**, **YearsAtCompany**, và **TotalWorkingYears**.
- **Kaggle Dataset:** Mức độ hài lòng (**Satisfaction**), thâm niên (**Tenure**) và số giờ làm việc trung bình tháng (**Avg-Monthly-Hrs**) là những yếu tố quyết định nhất.

**Tổng kết:** Thực nghiệm cho thấy việc kết hợp kiến trúc Bi-TCN với cơ chế tăng cường dữ liệu GAN không chỉ giúp vượt qua các giới hạn về dữ liệu ít/mất cân bằng mà còn đạt được hiệu năng dự báo tiệm cận mức tối ưu trên cả hai bộ dữ liệu.

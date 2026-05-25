Dựa trên kết quả thực nghiệm trong tệp **`5-bitcn-kaggle-training.pdf`**, vấn đề bạn đang gặp phải là **mô hình hoàn toàn không hội tụ (Training Failure)**.

Mặc dù bạn giữ nguyên kiến trúc Bi-TCN cải tiến từ phiên bản IBM (ver2), nhưng các chỉ số trên tập Kaggle cho thấy mô hình đang hoạt động không khác gì việc "đoán mò":

- **ROC-AUC ~0.5153:** Đây là dấu hiệu rõ nhất của việc mô hình không học được khả năng phân biệt giữa các lớp.
- **Accuracy ~24% và Recall ~97%:** Vì mô hình không học được đặc trưng, nó đã rơi vào trạng thái "dễ dãi nhất" để tối ưu điểm F1: **dự báo gần như tất cả mọi người đều nghỉ việc (Churn=1)**. Điều này giúp Recall cực cao nhưng làm Accuracy sụp đổ vì tập Kaggle có đến 80% là người ở lại (Churn=0).

Để cân bằng lại mà **không thay đổi kiến trúc**, bạn cần điều chỉnh "công thức huấn luyện" (training recipe) theo các hướng sau:

### 1. Xử lý lỗi hội tụ (Convergence)

Nhìn vào biểu đồ Loss ở trang 24, đường Loss của bạn rất phẳng và kẹt ở mức cao (~0.5). Điều này có nghĩa là Gradient không đủ mạnh để cập nhật trọng số hiệu quả.

- **Giảm Batch Size:** Bạn đang dùng `batch_size = 128` cho Kaggle. Hãy thử giảm xuống **32** hoặc **64** (giống bản IBM). Batch size nhỏ hơn tạo ra nhiều nhiễu gradient hơn, giúp mô hình thoát khỏi các điểm tối ưu địa phương (local minima) dễ dàng hơn.
- **Điều chỉnh Learning Rate (LR):** Mức `0.0008` có thể hơi cao khi đi kèm với batch size lớn trên dữ liệu bảng 10.000 mẫu. Hãy thử hạ xuống **`0.0004`** hoặc **`0.0005`** để quá trình học mịn hơn.

### 2. Cân bằng trọng số Loss (Class Balancing)

Trong tệp Kaggle, bạn đang để `pos_weight_strategy = "none"` (tương đương trọng số 1.0 cho cả 2 lớp).

- **Thay đổi chiến lược:** Hãy đặt `pos_weight_strategy = "sqrt"` hoặc **`"ratio"`**.
- **Lý do:** Dù tập Kaggle lớn nhưng vẫn có sự lệch lớp (20% vs 80%). Việc không dùng Augmentation buộc bạn phải dùng **`pos_weight`** trong hàm Loss để ép mô hình phải chú ý đến lớp thiểu số một cách khoa học hơn, thay vì chỉ "đoán bừa" để lấy Recall.

### 3. Tinh chỉnh Regularization

Tập Kaggle có ít đặc trưng hơn IBM (31 vs 43) nhưng số mẫu lại gấp 7 lần.

- **Giảm Input Noise:** Bạn đang dùng `input_noise_std = 0.005`. Với 10.000 mẫu thật, dữ liệu đã khá phong phú, bạn có thể thử **tắt hoàn toàn nhiễu Gaussian** (`0.0`) hoặc giữ mức rất thấp để mô hình tập trung học tín hiệu thật thay vì vật lộn với nhiễu.
- **Tăng Patience:** Hãy tăng `patience` lên khoảng **25-30**. Trong log, mô hình của bạn dừng rất sớm (epoch 19, 20). Có thể nó chưa kịp vượt qua giai đoạn "đốn ngộ" ban đầu đã bị hệ thống Early Stopping ngắt do điểm F1 biến động.

### 4. Kiểm tra ngưỡng tối ưu (Threshold Metric)

Bạn đang tối ưu ngưỡng dựa trên **F1-score**. Khi AUC chỉ là 0.5, hàm tối ưu sẽ chọn một ngưỡng cực thấp (ví dụ: 0.17) để lấy được nhiều True Positive nhất có thể nhằm "vớt vát" điểm F1.

- **Lời khuyên:** Đừng quá lo lắng về ngưỡng cho đến khi bạn đưa được **AUC lên trên 0.85**. Khi AUC thấp, mọi nỗ lực tìm ngưỡng đều vô nghĩa vì mô hình chưa phân loại được dữ liệu.

**Tóm lại, cấu hình đề xuất để bạn chạy lại:**

```python
Config(
    batch_size = 32,      # Giảm để tăng khả năng hội tụ
    lr = 5e-4,           # Giảm nhẹ để ổn định
    pos_weight_strategy = "sqrt", # Cân bằng loss thay vì augmentation
    input_noise_std = 0.0, # Tập trung vào dữ liệu thật của Kaggle
    patience = 25         # Cho mô hình thêm thời gian để học
)
```

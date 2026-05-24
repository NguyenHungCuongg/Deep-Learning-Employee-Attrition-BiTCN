## Dưới đây là đánh giá toàn diện về mô hình Bi-TCN (Bidirectional Temporal Convolutional Network) được triển khai trong file tài liệu huấn luyện để dự đoán sự nghỉ việc của nhân viên (Employee Attrition Prediction) (p. 1).

## 1. Điểm mạnh và Thiết kế Kiến trúc

- Kiến trúc sáng tạo và tối ưu cấu trúc: Việc kết hợp hai nhánh Bi-TCN song song với kích thước kernel khác nhau (Nhánh 1 dùng kernel=3, filter=16; Nhánh 2 dùng kernel=5, filter=32) giúp mô hình đồng thời bắt được cả đặc trưng cục bộ ngắn hạn và dài hạn (pp. 11, 13). Các tầng ẩn cũng đã được giảm số bộ lọc (filters) một cách chủ động để hạn chế hiện tượng quá khớp (overfitting) (p. 34).
- Tích hợp cơ chế Attention: Mô hình sử dụng AttentionLayer (Channel Attention) để tự động cân chỉnh và tái trọng số cho các đặc trưng quan trọng trước khi đưa vào phân loại, giúp tăng khả năng phân biệt (pp. 12-13).
- Chiến lược xử lý dữ liệu chuẩn mực:
- Quy trình kiểm định chéo 5-Fold Cross-Validation được thiết kế an toàn chống rò rỉ dữ liệu (leakage-safe), chỉ áp dụng tăng cường dữ liệu trên tập Train của từng fold (pp. 1, 14).
  - Sử dụng mạng phát sinh GAN (Generative Adversarial Networks) để cân bằng các lớp dữ liệu bị lệch (Class Imbalance) (pp. 1, 10).
  - Hàm tìm ngưỡng phân loại tối ưu (find_optimal_threshold) tập trung tối ưu chỉ số $F_2$-score, giúp ưu tiên tỷ lệ phát hiện thực tế (Recall) nhằm tránh bỏ sót những nhân viên có nguy cơ nghỉ việc (pp. 4-5).

---

## 2. Kết quả Hiệu năng (Metrics)

Dựa trên bảng tổng hợp kết quả đánh giá chéo (p. 25):

| Chỉ số (Metric) | Giá trị Trung bình (Mean ± Std) | Đánh giá tổng quan                                                                      |
| --------------- | ------------------------------- | --------------------------------------------------------------------------------------- |
| Accuracy        | $0.8259 \pm 0.0404$             | Khá cao và ổn định qua các fold (p. 25).                                                |
| AUC             | $0.7497 \pm 0.0309$             | Đạt mức khá, mô hình có khả năng phân biệt tốt giữa 2 nhóm (p. 25).                     |
| Precision       | $0.4814 \pm 0.0882$             | Thấp. Cứ khoảng 2 người mô hình đoán nghỉ việc thì chỉ có 1 người thực sự nghỉ (p. 25). |
| Recall          | $0.4633 \pm 0.0958$             | Thấp và biến động mạnh giữa các fold (từ 0.36 đến 0.56) (p. 25).                        |
| F1-Score        | $0.4624 \pm 0.0598$             | Mức trung bình do bị kéo xuống bởi cả Precision và Recall (p. 25).                      |

---

## 3. Vấn đề Tồn tại và Điểm Cần Cải thiện## Vấn đề Overfitting Nghiêm trọng khi Huấn luyện

Mô hình bị quá khớp rất nặng ngay từ các epoch đầu tiên. Nhìn vào log huấn luyện và đồ thị hàm mất mát (Loss Curves) (pp. 20, 31):

- Train Loss giảm xuống rất thấp (khoảng 0.06 đến 0.19) (pp. 20-22).
- Validation Loss lại tăng vọt lên rất cao (khoảng 0.77 đến 0.96), vượt xa mức loss thông thường (pp. 20-22, 31).
- Hệ thống Early Stopping liên tục bị kích hoạt rất sớm (tại epoch 19 - 26 trên tổng số 30 epoch) do không có sự cải thiện về hàm mất mát trên tập kiểm định (pp. 20-23).

## Mâu thuẫn trong Tăng cường Dữ liệu (Data Augmentation)

- Mặc dù tập dữ liệu huấn luyện sau khi chạy GAN đã được cân bằng hoàn hảo ($50\%$ lớp 0 và $50\%$ lớp 1 với 9176 mẫu) (p. 20), hiệu năng dự đoán nội bộ trên tập Train (Train Metrics) lại cực kỳ tệ với các chỉ số đều xấp xỉ 0.50 (bằng với đoán ngẫu nhiên) (pp. 20-22).
- Nguyên nhân: Có thể mạng GAN đang phát sinh ra các dữ liệu giả lập mang quá nhiều nhiễu ngẫu nhiên, hoàn toàn phá vỡ cấu trúc phân phối đặc trưng của dữ liệu gốc, khiến mô hình Bi-TCN không thể học được gì từ tập dữ liệu tăng cường này.

## Độ lệch Ngưỡng phân loại (Decision Boundary)

- Ngưỡng tối ưu tìm được dao động rất thấp từ 0.08 đến 0.22 (trung bình 0.162) thay vì mức 0.5 mặc định (pp. 23, 25).
- Biểu đồ phân phối dự đoán (Prediction Distribution) cho thấy mô hình bị nén xác suất (p. 38): điểm trung bình đoán cho nhóm nghỉ việc thực tế chỉ đạt 0.3402 (p. 38). Điều này chứng tỏ mô hình đang cực kỳ "thiếu tự tin" khi đưa ra các dự đoán mang nhãn nghỉ việc (lớp 1).

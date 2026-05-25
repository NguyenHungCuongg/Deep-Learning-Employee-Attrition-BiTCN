# Tổng hợp Hướng dẫn Xây dựng Mô hình Bi-TCN (Theo Bài báo Gốc)

## 1. Kiến trúc Tổng thể (Overall Architecture)
Mô hình đề xuất là một mạng **Ensemble Deep Learning** dựa trên cấu trúc **Mạng Chập Thời Gian Hai Chiều (Bi-TCN)**. Cấu trúc này được thiết kế để nắm bắt các mối tương quan phức tạp trong dữ liệu nhân sự thông qua việc xử lý song song.

### Luồng xử lý dữ liệu (Figure 2):
1.  **Input Layer**: Tiếp nhận các đặc trưng bảng (Tabular features) đã chuẩn hóa.
2.  **Initial Dense Block**: 
    *   **Fully Connected Layer**: 128 đơn vị (units).
    *   **Batch Normalization Layer**: Chuẩn hóa dữ liệu trước khi đi vào các khối chập.
3.  **Parallel Bi-TCN Ensemble**: Dữ liệu từ tầng BN được chia thành hai nhánh chạy song song:
    *   **Nhánh 1**: Bi-TCN với 32 bộ lọc (filters), hạt nhân (kernel size) bằng 3.
    *   **Nhánh 2**: Bi-TCN với 64 bộ lọc (filters), hạt nhân (kernel size) bằng 5.
4.  **Concatenate Layer**: Gộp kết quả từ hai nhánh song song [Figure 2].
5.  **Output Processing**:
    *   **Flatten Layer**: Làm phẳng dữ liệu [Figure 2].
    *   **Softmax Layer**: Phân loại nhị phân (Nghỉ việc/Ở lại) [Figure 2, 357].

---

## 2. Chi tiết Khối Bi-TCN (Bi-TCN Block Components)
Mỗi khối TCN trong hai nhánh trên được cấu tạo từ các thành phần chính để đảm bảo tính ổn định và khả năng học phụ thuộc xa (Figure 3):

*   **Dilated Causal Convolutions**: Chập nhân quả có giãn cách giúp mở rộng vùng tiếp nhận (receptive field) mà không tăng tham số. 
    *   **Dilation rates**: Cả hai tầng đều sử dụng danh sách giãn cách ``.
*   **Bidirectionality**: Xử lý dữ liệu theo cả hai chiều tiến (forward) và lùi (backward) để phân tích toàn diện các quy luật.
*   **Residual Connections**: Các kết nối tắt (skip connections) cộng đầu vào với đầu ra của khối chập để tránh triệt tiêu đạo hàm.
*   **Thành phần bổ trợ**: Bao gồm **Weight Normalization**, **ReLU Activation**, và **Dropout** (tỷ lệ 0.5) bên trong mỗi khối [336, 337, Figure 3].

---

## 3. Siêu tham số Huấn luyện (Training Hyperparameters)
Dựa trên cấu hình thực nghiệm tối ưu của bài báo (Table 4):

*   **Optimizer**: Adam với tốc độ học (learning rate) = 0.001.
*   **Loss Function**: Binary Crossentropy.
*   **Batch Size**: 128.
*   **Epochs**: 50.
*   **Dropout Rate**: 0.5 (áp dụng sau các khối Bi-TCN).
*   **Regularization**: L2 Weight Decay (để ngăn ngừa quá khớp).

---

## 4. Quy trình Tiền xử lý & Tăng cường (Preprocessing & Augmentation)
*   **GAN-based Augmentation**: Sử dụng GAN để cân bằng bộ dữ liệu IBM (từ 1,470 mẫu lên tổng cộng 3,000 mẫu: 1,500 "Nghỉ việc" và 1,500 "Ở lại").
*   **Leakage-safe Cross-Validation**: Áp dụng kiểm chéo 5-fold, trong đó việc tăng cường dữ liệu và chuẩn hóa (Normalization) **chỉ được thực hiện trên tập huấn luyện** của từng fold.
*   **Normalization**: Đưa tất cả các biến số về phạm vi `` trước khi đưa vào mô hình.


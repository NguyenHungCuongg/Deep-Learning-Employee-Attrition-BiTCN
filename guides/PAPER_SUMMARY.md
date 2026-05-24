Đây là bản "giải mã" chi tiết các thông số kỹ thuật và sơ đồ từ file applsci-15-02984.pdf sang dạng văn bản thuần để AI có thể xử lý chính xác.

---

### PHẦN 1: THÔNG SỐ SIÊU THAM SỐ (HYPERPARAMETERS)
*Dựa trên Table 4 trang 18 của bài báo:*

*   **Optimizer:** Adam (Learning rate = 0.001).
*   **Loss function:** Binary Crossentropy.
*   **Batch Size:** 128.
*   **Epochs:** 50.
*   **Dropout Rate:** 0.5.
*   **Weight Decay (L2):** Phạt các trọng số lớn để tránh overfitting.
*   **Bi-TCN Layer 1:** 32 filters, Kernel size = 3, Dilations = ****.
*   **Bi-TCN Layer 2:** 64 filters, Kernel size = 5, Dilations = ****.
*   **Fully Connected Layer (đầu mạng):** 128 units.

---

### PHẦN 2: KIẾN TRÚC MÔ HÌNH CHI TIẾT (MODEL ARCHITECTURE)
*Dựa trên Figure 2 trang 8 và mô tả tại trang 7-9:*

1.  **Input Layer:** Nhận dữ liệu bảng đã chuẩn hóa (43 đặc trưng cho bộ IBM).
2.  **Initial Dense Block:**
    *   Dữ liệu đi qua một tầng **Fully Connected (128 units)**.
    *   Sau đó đi qua tầng **Batch Normalization** để chuẩn hóa dữ liệu trước khi vào các khối chập.
3.  **Parallel Bi-TCN Ensemble (Cấu trúc song song):** Đầu ra từ tầng BN ở trên sẽ được chia làm 2 nhánh chạy song song:
    *   **Nhánh 1:** Đi vào khối Bi-TCN với 32 bộ lọc, hạt nhân cỡ 3.
    *   **Nhánh 2:** Đi vào khối Bi-TCN với 64 bộ lọc, hạt nhân cỡ 5.
    *   Cả hai nhánh đều phải xử lý dữ liệu theo hai chiều (Bidirectional - tiến và lùi).
4.  **Concatenation Layer:** Gộp (concatenate) kết quả đầu ra của hai nhánh song song này lại với nhau.
5.  **Output Processing:**
    *   **Flatten Layer:** Làm phẳng dữ liệu.
    *   **Softmax Layer:** Đưa ra xác suất dự báo cho 2 lớp (Attrition vs No Attrition).
    *   **Dropout:** Áp dụng tỷ lệ 0.5 sau mỗi khối Bi-TCN.

---

### PHẦN 3: CẤU TRÚC CHI TIẾT CỦA MỘT KHỐI RESIDUAL BLOCK (TCN)
*Dựa trên Figure 3 trang 9:*

Mỗi khối Bi-TCN nhỏ bên trong phải chứa:
*   **Dilated Causal Convolution:** Chập nhân quả có giãn cách (đảm bảo không rò rỉ dữ liệu tương lai).
*   **Weight Normalization:** Chuẩn hóa trọng số.
*   **ReLU Activation:** Hàm kích hoạt.
*   **Dropout Layer.**
*   **Residual Connection:** Một kết nối tắt cộng đầu vào ban đầu với đầu ra sau khi chập (nếu số kênh thay đổi, dùng 1x1 Conv để khớp kích thước).

---

### PHẦN 4: LỆNH YÊU CẦU AGENTIC AI

> "Dựa trên các thông số kỹ thuật tôi vừa cung cấp ở trên, hãy thực hiện các việc sau:
> 1. Sửa lại Class `BiTCN` trong code của tôi. Thay vì xếp chồng (stack) `bitcn1` rồi đến `bitcn2`, hãy chuyển chúng thành hai nhánh song song và dùng `torch.cat` để gộp đầu ra.
> 2. Đưa tầng `nn.Linear(input_dim, 128)` và `nn.BatchNorm1d` lên đầu hàm `forward`, ngay sau khi nhận input và trước khi chia nhánh Bi-TCN.
> 3. Cập nhật danh sách dilation thành `` cho cả hai tầng Bi-TCN thay vì chỉ có 1 và 2 như hiện tại.
> 4. Đảm bảo hàm `forward` xử lý đúng chiều dữ liệu của `Conv1d` (Batch, Channels, Length)."

---

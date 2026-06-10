# IR Course Project - Hệ thống tìm kiếm và phân tích cảm xúc đánh giá nhà hàng

Đây là đồ án môn **Truy hồi thông tin** với đề tài xây dựng hệ thống tìm kiếm và khai phá cảm xúc khách hàng dựa trên dữ liệu đánh giá nhà hàng.

Project sử dụng dữ liệu review từ Yelp để xây dựng hệ thống tìm kiếm nhà hàng, kết hợp giữa tìm kiếm theo từ khóa, tìm kiếm ngữ nghĩa và phân tích cảm xúc của người dùng.

---

## Mục tiêu project

Project tập trung vào các phần chính:

* Xử lý dữ liệu đánh giá nhà hàng từ Yelp
* Xây dựng hệ thống tìm kiếm review và nhà hàng
* Kết hợp BM25 và FAISS để cải thiện kết quả tìm kiếm
* Phân loại cảm xúc đánh giá thành Positive, Neutral, Negative
* Phân tích cảm xúc theo từng khía cạnh như đồ ăn, phục vụ, không gian, giá cả
* Xây dựng giao diện dashboard bằng Streamlit

---

## Chức năng chính

* Tiền xử lý dữ liệu review nhà hàng
* Làm sạch văn bản đánh giá
* Tạo chỉ mục tìm kiếm bằng BM25
* Tạo chỉ mục tìm kiếm ngữ nghĩa bằng FAISS
* Kết hợp kết quả bằng Hybrid Search
* Phân loại cảm xúc bằng LinearSVC
* Phân tích cảm xúc theo khía cạnh bằng VADER và từ điển khía cạnh
* Gợi ý nhà hàng dựa trên truy vấn và điểm cảm xúc
* Hiển thị kết quả trên giao diện Streamlit
* Đánh giá mô hình truy hồi và mô hình phân loại cảm xúc

---

## Công nghệ sử dụng

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* BM25
* FAISS
* Sentence Transformers
* VADER Sentiment
* Matplotlib / Plotly
* Pytest

---

## Mô hình và phương pháp

### 1. BM25

BM25 được dùng để tìm kiếm theo từ khóa.
Cách này phù hợp khi người dùng nhập đúng tên món ăn, tên nhà hàng hoặc các từ khóa xuất hiện trực tiếp trong review.

Ví dụ:

```text
good pizza
cheap restaurant
friendly staff
```

### 2. FAISS Semantic Search

FAISS được dùng cho tìm kiếm ngữ nghĩa.
Các review được mã hóa thành vector bằng mô hình sentence embedding, sau đó tìm các review có ý nghĩa gần với truy vấn của người dùng.

Cách này giúp hệ thống hiểu các câu có nghĩa gần nhau, dù không trùng từ khóa hoàn toàn.

Ví dụ:

```text
tasty food
delicious meal
good flavor
```

### 3. Hybrid Search

Hệ thống kết hợp BM25 và FAISS để tận dụng cả hai hướng:

* BM25: mạnh về tìm kiếm từ khóa chính xác
* FAISS: mạnh về tìm kiếm theo ngữ nghĩa
* RRF: dùng để trộn và xếp hạng lại kết quả

Nhờ vậy kết quả tìm kiếm ổn định hơn so với chỉ dùng một phương pháp riêng lẻ.

### 4. Sentiment Analysis

Project sử dụng mô hình Machine Learning để phân loại cảm xúc review:

* Positive
* Neutral
* Negative

Các mô hình được thử nghiệm gồm:

* Multinomial Naive Bayes
* Logistic Regression
* LinearSVC

Trong đó, LinearSVC được chọn làm mô hình chính vì cho kết quả tốt và phù hợp với dữ liệu văn bản dạng TF-IDF.

### 5. ABSA

ABSA là phân tích cảm xúc theo từng khía cạnh.
Trong project này, hệ thống chia review thành các nhóm khía cạnh như:

* Food
* Service
* Ambience
* Price

Sau đó dùng VADER để chấm điểm cảm xúc cho từng nhóm, giúp người dùng biết nhà hàng mạnh hoặc yếu ở điểm nào.

---

## Cấu trúc thư mục

```text
IR_Course_Project/
│
├── .streamlit/                 # Cấu hình Streamlit
├── evaluation/                 # Dữ liệu hoặc file đánh giá
├── src/
│   ├── app/                    # Giao diện dashboard Streamlit
│   ├── analytics/              # Xử lý thống kê, phân tích
│   ├── data_ingest/            # Đọc và chuẩn bị dữ liệu
│   ├── evaluation/             # Đánh giá mô hình
│   ├── indexing/               # Tạo chỉ mục BM25, FAISS
│   ├── preprocessing/          # Làm sạch dữ liệu văn bản
│   ├── retrieval/              # Xử lý tìm kiếm
│   ├── sentiment/              # Phân tích cảm xúc
│   └── utils/                  # Hàm tiện ích
│
├── tests/                      # Unit test
├── Makefile                    # Lệnh chạy nhanh
├── requirements.txt            # Danh sách thư viện
└── README.md
```

---

## Cách cài đặt

### 1. Clone project

```bash
git clone https://github.com/Phucht59/IR_Course_Project.git
cd IR_Course_Project
```

### 2. Tạo môi trường ảo

Trên Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Trên macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài thư viện

```bash
pip install -r requirements.txt
```

---

## Cách chạy project

Chạy dashboard Streamlit:

```bash
streamlit run src/app/dashboard.py
```

## Cách chạy bằng Makefile

Project có hỗ trợ một số lệnh nhanh bằng Makefile.

Cài thư viện:

```bash
make install
```

Chạy test:

```bash
make test
```

Chạy đánh giá:

```bash
make eval
```

---

## Quy trình xử lý chính

```text
Yelp Dataset
    ↓
Lọc dữ liệu nhà hàng
    ↓
Làm sạch văn bản review
    ↓
Tạo chỉ mục BM25
    ↓
Tạo vector embedding và FAISS index
    ↓
Huấn luyện mô hình sentiment
    ↓
Phân tích cảm xúc theo khía cạnh
    ↓
Tìm kiếm lai BM25 + FAISS
    ↓
Hiển thị kết quả trên Streamlit Dashboard
```

---

## Kết quả thực nghiệm

Trong quá trình thử nghiệm, project đánh giá cả phần tìm kiếm và phân loại cảm xúc.

Với bài toán phân loại cảm xúc 3 lớp, các mô hình được so sánh gồm Naive Bayes, Logistic Regression và LinearSVC.
LinearSVC cho kết quả tốt và được chọn làm mô hình chính.

Kết quả nổi bật:

* LinearSVC đạt accuracy khoảng 86% với bài toán 3 lớp
* Với bài toán 2 lớp Positive / Negative, LinearSVC đạt kết quả cao hơn
* Hybrid Search giúp cải thiện khả năng tìm kiếm so với chỉ dùng từ khóa
* ABSA giúp phân tích sâu hơn theo từng khía cạnh thay vì chỉ đánh giá cảm xúc chung

---

## Giao diện hệ thống

Dashboard được xây dựng bằng Streamlit, gồm các phần chính:

* Ô nhập truy vấn tìm kiếm
* Kết quả nhà hàng phù hợp
* Thông tin review liên quan
* Điểm cảm xúc
* Phân tích theo khía cạnh
* Biểu đồ trực quan hỗ trợ người dùng xem nhanh kết quả

---

## Dataset

Project sử dụng dữ liệu đánh giá từ Yelp Open Dataset.

Trong quá trình xử lý, dữ liệu được lọc lại để tập trung vào nhóm nhà hàng.
Các review được làm sạch, chuẩn hóa và gán nhãn cảm xúc dựa trên số sao.

Cách gán nhãn:

```text
1 - 2 sao: Negative
3 sao: Neutral
4 - 5 sao: Positive
```

---

## Ghi chú

* Project hiện tập trung trên dữ liệu tiếng Anh.
* Dữ liệu dùng trong project là tập con của Yelp Dataset.
* Kết quả tìm kiếm phụ thuộc vào chất lượng dữ liệu và câu truy vấn.
* Phần ABSA đang xử lý theo hướng rule-based nên vẫn có thể cải thiện thêm.
* Project phù hợp để học về Information Retrieval, NLP và Sentiment Analysis.

---

## Hướng phát triển

Một số hướng có thể phát triển thêm:

* Cải thiện mô hình cảm xúc bằng BERT hoặc các mô hình Deep Learning
* Mở rộng dữ liệu lớn hơn
* Hỗ trợ tiếng Việt
* Cải thiện ABSA bằng mô hình học sâu
* Thêm chức năng đăng nhập người dùng
* Lưu lịch sử tìm kiếm
* Cá nhân hóa gợi ý nhà hàng
* Deploy dashboard lên cloud hoặc Hugging Face Spaces

---

## Tác giả

**Trần Hoàng Phúc**

GitHub: [Phucht59](https://github.com/Phucht59)

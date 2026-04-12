import faiss
import pickle
from scipy import sparse
import os

# Tự động tìm đường dẫn gốc của Project (review_IR) dù chạy từ đâu
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

print("\n" + "="*50)
print("🔍 KIỂM TRA BÊN TRONG CÁC FILE KNOWLEDGE BASE")
print("="*50)

# 1. FAISS
print("\n[1] BÊN TRONG FAISS (faiss_index.bin)")
try:
    faiss_path = os.path.join(PROJECT_ROOT, 'models', 'faiss_index.bin')
    idx = faiss.read_index(faiss_path)
    print(f"  ➜ Tổng số Vector (Bài đánh giá): {idx.ntotal:,.0f}")
    print(f"  ➜ Số chiều không gian (Dimensions): {idx.d} chiều")
    print(f"  ➜ Trạng thái huẩn luyện: {'Đã Train' if idx.is_trained else 'Chưa Train'}")
except Exception as e:
    print(f"Lỗi đọc FAISS: {e}")

# 2. BM25
print("\n[2] BÊN TRONG BM25 (bm25_index.pkl)")
try:
    bm25_path = os.path.join(PROJECT_ROOT, 'models', 'bm25_index.pkl')
    with open(bm25_path, 'rb') as f:
        bm25 = pickle.load(f)
    print(f"  ➜ Tổng số Document (Bài đánh giá): {bm25.corpus_size:,.0f}")
    print(f"  ➜ Chiều dài trung bình mỗi Document: {bm25.avgdl:.2f} từ")
    sample_vocab = list(bm25.doc_freqs[0].items())[:5]
    print(f"  ➜ Ví dụ từ khóa và tần suất ở Doc 1: {sample_vocab}")
except Exception as e:
    print(f"Lỗi đọc BM25: {e}")

# 3. TF-IDF
print("\n[3] BÊN TRONG TF-IDF Baseline (tfidf_matrix.npz)")
try:
    tfidf_path = os.path.join(PROJECT_ROOT, 'models', 'tfidf_matrix.npz')
    mat = sparse.load_npz(tfidf_path)
    print(f"  ➜ Cấu hình (Documents x Tính từ): {mat.shape}")
    print(f"  ➜ Tổng số giá trị có nghĩa (Non-zero elements): {mat.nnz:,.0f}")
    print("  ➜ Kiểu cấu trúc ảo: Compressed Sparse Row (CSR)")
except Exception as e:
    print(f"Lỗi đọc TF-IDF: {e}")

print("\n(Các file này là dạng nhị phân/ma trận để máy tính đọc cực nhanh, không thể mở bằng Notepad thông thường)\n")

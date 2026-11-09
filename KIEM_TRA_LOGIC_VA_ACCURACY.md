# BÁO CÁO KIỂM TRA TOÀN DIỆN LOGIC PIPELINE & NGUYÊN NHÂN ĐỘ CHÍNH XÁC THẤP

> **Dự án:** `video-visual-rag` (AIC-RAG)  
> **Mục đích:** Đánh giá độ đúng đắn của logic xử lý, kiểm tra tính đồng bộ của các embedding models, phân tích căn nguyên tại sao độ chính xác tìm kiếm/retrieval bị thấp và đề xuất giải pháp tối ưu.  
> **Trạng thái Code:** Giữ nguyên toàn bộ mã nguồn hiện tại, không can thiệp sửa đổi trực tiếp vào code.

---

## 1. TỔNG QUAN ĐIỀU HÀNH (EXECUTIVE SUMMARY)

Sau khi rà soát chi tiết toàn bộ codebase từ Phase 1 (Keyframe Extraction) đến Phase 4 (Hybrid Search, Rerank, Frame Selection, TRAKE), kết luận tổng quan là:
* **Về mặt kỹ thuật / Run-time:** Pipeline chạy thông suốt, không crash, các luồng xử lý dữ liệu và singleton model được thiết kế cẩn thận để tránh OOM trên GPU Kaggle T4.
* **Về mặt logic AI / RAG & Độ chính xác:** Hệ thống đang gặp phải **6 lỗ hổng logic và kiến trúc cực kỳ nghiêm trọng**. Trong đó, lỗ hổng lớn nhất là **sự ngắt kết nối hoàn toàn về thị giác (Visual Blindness) trong khâu Retrieval** và **lỗi toán học khi đặt ngưỡng điểm RRF**.
* **Bản chất hiện tại:** Mặc dù dự án mang tên *Video Visual RAG*, nhưng thực chất ở khâu tìm kiếm (Retrieval) nó đang hoạt động như một công cụ **"Text Search trên Tiêu đề & Chữ OCR"**. Mọi thông tin về màu sắc, con người, hành động, sự vật, bối cảnh hình ảnh của video **hoàn toàn không được đưa vào chỉ mục vector** để tìm kiếm!

---

## 2. KIỂM TRA EMBEDDING MODEL: CÓ CÙNG MODEL KHÔNG?

Câu trả lời gồm 2 vế: **CÙNG ở tầng Text, nhưng LỆCH PHA HOÀN TOÀN ở tầng Multimodal (Ảnh vs Chữ).**

### 2.1. Tầng Text-to-Text: Có đồng bộ không?
* **Model sử dụng:** `BAAI/bge-m3` (FlagEmbedding).
* **Dense vector:** Cả tài liệu (`full_text_for_embedding` trong `embedder.py`) và câu truy vấn của người dùng (`search_qdrant_hybrid` trong `Qdrant_index.py`) đều được sinh ra bởi cùng một hàm `embed_texts()` dùng `BGEM3FlagModel` với dimension = 1024, chuẩn hóa cosine.
* **Sparse vector:** Cả hai bên đều dùng chung `lexical_weights` (token weights) trích xuất từ BGE-M3.
* **Kết luận:** Giữa câu hỏi dạng text và dữ liệu văn bản đã index, **embedding model là hoàn toàn đồng bộ (100% cùng model)**.

### 2.2. Tầng Vision-to-Text: Lỗ hổng "Đứt gãy không gian vector"
* Trong dự án có xuất hiện model Multimodal: **SigLIP 2** (`google/siglip2-so400m-patch14-naflex`, dimension = 1152) tại `src/phase2_captioning/visual_embedder.py`.
* **Tuy nhiên, SigLIP KHÔNG ĐƯỢC DÙNG ĐỂ INDEX VÀO QDRANT!**
  * Qdrant collection chỉ cấu hình lưu 1 vector Dense (dim 1024 của BGE-M3) và 1 vector Sparse (BGE-M3).
  * Không có bất kỳ vector hình ảnh (SigLIP image embedding) nào được lưu vào Qdrant.
* **Hậu quả:**
  * BGE-M3 (1024 chiều) chỉ hiểu văn bản, **không thể nhúng ảnh**.
  * SigLIP 2 (1152 chiều) nhúng được cả ảnh và text vào cùng một không gian, **nhưng lại không được dùng để tìm kiếm segment**.
  * Hai model này có cấu trúc, số chiều và không gian biểu diễn hoàn toàn khác nhau.
  * SigLIP chỉ được gọi ở bước cuối cùng (`frame_selector.py`) để chọn 1 frame trong 3-5 frame của segment *sau khi segment đó đã được tìm ra*. Nếu segment tìm sai ngay từ đầu, SigLIP hoàn toàn vô dụng ("Garbage In, Garbage Out").

---

## 3. NGUYÊN NHÂN CỐT LÕI KHIẾN ĐỘ CHÍNH XÁC (ACCURACY/RECALL) KHÁ THẤP

### 🔴 Lỗ hổng 1: Phase 2 bị "Mù thị giác" hoàn toàn (The OCR-Only Fallacy)
* **Vấn đề trong code:**
  Tại `src/phase2_captioning/metadata_builder.py` và `ocr_engine.py`:
  ```python
  ocr_text, visual_caption = build_segment_caption(keyframe_paths, predictor=ocr_predictor)
  # Trong đó visual_caption được sinh bởi build_caption_from_ocr():
  # Nếu có OCR text: "On-screen text detected: " + "; ".join(lines) + "."
  # Nếu KHÔNG có OCR text: "No on-screen text detected in this segment."
  ```
  Và chuỗi đưa vào BGE-M3 để nhúng (`full_text_for_embedding`):
  ```python
  full_text = f"TITLE: {video_title} | VISUAL: {visual_caption} | OCR: {ocr_text}"
  ```
* **Thực tế diễn ra:**
  * Nếu video là cảnh đời thực, thể thao, flycam, vlog... không có phụ đề hoặc chữ viết trên khung hình, thì `VISUAL` sẽ là chuỗi vô nghĩa: `"No on-screen text detected in this segment."`, còn `OCR` là rỗng `""`.
  * Chuỗi nhúng lúc này chỉ còn vỏn vẹn: `TITLE: <tên video> | VISUAL: No on-screen text detected in this segment. | OCR: `.
  * Khi người dùng nhập truy vấn thị giác: *"Người đàn ông mặc áo xanh lái xe"* hoặc *"Cảnh hoàng hôn trên biển"*:
    * BGE-M3 sẽ lấy vector của câu hỏi này so khớp với câu `"No on-screen text..."` và tiêu đề video.
    * Do hoàn toàn không có từ khóa về màu áo, con người, cảnh quan, điểm tương đồng ngữ nghĩa sẽ cực thấp hoặc rơi vào ngẫu nhiên.
  * **Đó là lý do lớn nhất khiến Recall của hệ thống gần như bằng 0 đối với các truy vấn miêu tả hình ảnh/hành động.**

---

### 🔴 Lỗ hổng 2: Lỗi toán học nghiêm trọng ở bộ lọc điểm RRF (`min_relevance_score: 0.15`)
* **Vấn đề trong code:**
  * Trong `config/settings.yaml` (dòng 126): `min_relevance_score: 0.15`.
  * Trong `src/search_engine/hybrid_search.py` (dòng 64-65):
    ```python
    for seg_id, score in raw_results:
        # Qdrant's RRF score is not normalized to [0,1], but we still apply the threshold
        if score < min_relevance_score:
            continue
        scored.append(...)
    ```
* **Bản chất toán học của RRF (Reciprocal Rank Fusion):**
  Qdrant tính điểm RRF theo công thức chuẩn:
  $$\text{Score}_{\text{RRF}}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + \text{rank}_m(d)}$$
  * Nếu một segment đứng **Hạng 1** ở cả Dense và Sparse:
    $$\text{Score}_{\max} = \frac{1}{60 + 1} + \frac{1}{60 + 1} = \frac{1}{61} + \frac{1}{61} \approx 0.01639 + 0.01639 = \mathbf{0.0328}$$
  * Nếu segment chỉ lọt Top 1 ở Dense, rớt khỏi Top Sparse:
    $$\text{Score} = \frac{1}{61} \approx \mathbf{0.0164}$$
  * **ĐIỂM RRF CỦA QDRANT TỐI ĐA CHỈ ĐẠT ~0.0328, KHÔNG BAO GIỜ VƯỢT QUÁ 0.033!**
* **Hậu quả:**
  * Ngưỡng `min_relevance_score` lại đặt là **`0.15`**!
  * Vì `0.0328 < 0.15`, điều kiện `if score < min_relevance_score: continue` **SẼ LỌC VỨT BỎ TOÀN BỘ KẾT QUẢ** (hoặc nếu Qdrant có biến thể normalize thì cũng làm mất 80-90% ứng viên tiềm năng).
  * *Nguồn gốc bug:* Trước đây codebase dùng FAISS Cosine Similarity (thang điểm từ 0 đến 1, ngưỡng 0.15 là hợp lý). Khi nâng cấp sang Qdrant Native RRF, người viết đã giữ nguyên ngưỡng 0.15 mà không cập nhật lại theo miền giá trị của hàm RRF.

---

### 🔴 Lỗ hổng 3: VLM Reranker bị "mù ảnh" và lỗi Parse Regex làm đảo lộn điểm
* **Vấn đề 1: Reranker không hề nhìn thấy ảnh:**
  Tại `src/search_engine/reranker.py` (dòng 90):
  ```python
  raw = engine.generate(text_prompt=prompt, image_paths=None, max_new_tokens=150, temperature=0.0)
  ```
  Reranker gửi `image_paths=None` cho Qwen2.5-VL. Nội dung duy nhất Qwen đọc được là text tóm tắt:
  `VISUAL: {meta.get('visual_caption', '')} | OCR: {meta.get('ocr_screen_text', '')}`.  
  Như đã chỉ ra ở Lỗ hổng 1, `visual_caption` chỉ là text OCR lặp lại. Qwen hoàn toàn không thấy ảnh thực tế, nên việc chấm điểm chỉ là đoán mò dựa trên vài chữ OCR.
* **Vấn đề 2: Lỗi Regex bóc tách điểm (`re.findall`) làm tráo đổi điểm số:**
  Tại `reranker.py` (dòng 102-109):
  ```python
  numbers = re.findall(r"[-+]?\d*\.\d+|\d+", raw)
  scores = [float(s) for s in numbers]
  if len(scores) > len(batch):
      scores = scores[:len(batch)]
  ```
  * Prompt yêu cầu: Chấm điểm cho từng ứng viên `[0]`, `[1]`, `[2]`.
  * Nếu Qwen trả lời có kèm giải thích, ví dụ:  
    `"Candidate [0] score: 8, Candidate [1] score: 3"`
  * Hàm Regex `re.findall` sẽ nhặt ra danh sách: `['0', '8', '1', '3']`.
  * Cắt lấy `scores[:len(batch)]` (với batch=2) sẽ lấy được `[0.0, 8.0]`.
  * **Hậu quả:** Số thứ tự của ứng viên (`0` và `1`) bị lấy nhầm làm điểm số! Ứng viên số 0 bị chấm 0 điểm, ứng viên số 1 bị chấm 8 điểm! Toàn bộ thứ hạng bị đảo lộn hoàn toàn ngẫu nhiên.

---

### 🔴 Lỗ hổng 4: Hardcoded `limit=20` trong Qdrant & Lỗi lọc theo Video ID
* **Vấn đề trong code:**
  Tại `src/phase3_indexing/Qdrant_index.py` (dòng 168, 173):
  ```python
  prefetch = [
      models.Prefetch(
          query=dense_query.tolist(),
          using="dense",
          limit=20,  # <-- CỨNG 20
      ),
      models.Prefetch(
          query=sparse_query,
          using="sparse",
          limit=20,  # <-- CỨNG 20
      ),
  ]
  ```
* **Hậu quả:**
  1. Dù bên ngoài truyền `final_top_k=50` hay `100`, Qdrant prefetch vẫn chỉ lấy tối đa 20 kết quả từ mỗi nhánh.
  2. Trong tính năng **TRAKE** hoặc lọc theo video (`--video-id`):
     * Code thực hiện tìm kiếm trên toàn bộ corpus rồi dùng Python filter: `[s for s in segments if s.metadata.get("video_id") == video_id]`.
     * Vì prefetch chỉ lấy top 20 của toàn bộ dataset, nếu các segment của video chỉ định nằm từ hạng 21 trở đi trên toàn corpus, sau khi filter danh sách sẽ **rỗng hoàn toàn (0 kết quả)**!
     * Lẽ ra phải đẩy điều kiện lọc vào `query_points` thông qua Qdrant Payload Filter:
       `query_filter=models.Filter(must=[models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id))])`.

---

### 🟡 Lỗ hổng 5: Mật độ trích xuất Keyframe quá thưa (Phase 1)
* **Thông số trong `settings.yaml`:**
  * `segment_window_sec: 20` (Độ dài segment: 15-30 giây).
  * `max_keyframes_per_segment: 5`, `min_keyframes_per_segment: 3`.
  * `ssim_threshold: 0.86`.
* **Hệ quả thực tế:**
  * Một phân đoạn video dài 20 đến 30 giây nhưng chỉ giữ lại từ 3 đến 5 khung hình.
  * Nghĩa là trung bình **4 đến 7 giây mới có 1 khung hình**.
  * Trong các video có tiết tấu nhanh, các hành động tức thời (ví dụ: ngã xe, giơ tay, ném bóng, xuất hiện logo trong 1-2 giây) rất dễ bị bỏ qua hoàn toàn.

---

### 🟡 Lỗ hổng 6: Khả năng hiểu tiếng Việt của SigLIP 2
* Model `google/siglip2-so400m-patch14-naflex` được train chủ yếu trên WebLI (tiếng Anh chiếm tỉ trọng lớn).
* Khi người dùng nhập câu truy vấn tiếng Việt vào `frame_selector.py`, khả năng căn chỉnh không gian Text-Image của SigLIP đối với tiếng Việt sẽ kém chính xác hơn nhiều so với tiếng Anh, dẫn đến việc chọn sai keyframe trong segment.

---

## 4. BẢNG TỔNG HỢP SO SÁNH VÀ ĐỐI CHIẾU CÁC THÀNH PHẦN

| Thành phần | Hiện trạng trong code | Đánh giá logic | Ảnh hưởng tới độ chính xác |
| :--- | :--- | :--- | :--- |
| **Indexing Model** | `BAAI/bge-m3` (Dense 1024 + Sparse) | Đúng kỹ thuật cho Text | Không nhúng được ảnh, mất thông tin thị giác |
| **Multimodal Model** | `SigLIP 2` (1152) | **Chỉ dùng lúc chọn frame cuối**, không index vào DB | Không tham gia tìm kiếm segment, vô dụng nếu search sai |
| **Visual Caption (Phase 2)** | Lấy trực tiếp từ OCR (`build_caption_from_ocr`) | **Sai lệch bản chất** | Video không chữ = caption rỗng. Gây trượt 100% visual queries |
| **RRF Fusion Threshold** | `min_relevance_score: 0.15` | **Lỗi toán học** | RRF max chỉ đạt 0.033. Ngưỡng 0.15 lọc bỏ hết kết quả |
| **VLM Reranker (`reranker.py`)** | Text-only + Regex `\d+` trích xuất điểm | **Lỗi logic nghiêm trọng** | Nhầm số thứ tự thành điểm, tráo đổi rank của ứng viên |
| **Qdrant Prefetch Limit** | Hardcode `limit=20` | Thiếu linh hoạt | Cắt cụt candidate pool, làm hỏng chức năng tìm theo `video_id` / TRAKE |
| **Keyframe Density** | 3-5 frames / 20-30s | Quá thưa | Dễ bỏ sót các khoảnh khắc ngắn |

---

## 5. ĐỀ XUẤT PHƯƠNG ÁN CẢI THIỆN TRIỆT ĐỂ (ACTIONABLE ROADMAP)

Để đưa hệ thống đạt độ chính xác cao và đúng nghĩa là **Video Visual RAG**, cần thực hiện các cải tiến theo thứ tự ưu tiên sau:

### Giai đoạn 1: Sửa ngay các lỗi logic toán học & code (Không tốn thêm chi phí tính toán)
1. **Sửa ngưỡng điểm RRF trong `hybrid_search.py`:**
   * Hạ `min_relevance_score` xuống `0.0` (hoặc `0.005` đối với RRF), hoặc chuẩn hóa điểm RRF về khoảng $[0, 1]$ trước khi filter:
     $$\text{Normalized\_Score} = \frac{\text{score}}{\text{max\_possible\_score}}$$
2. **Sửa Regex bóc tách điểm trong `reranker.py`:**
   * Sử dụng parser JSON an toàn (`_extract_json_array`) đã có sẵn thay vì `re.findall(r"\d+")`.
   * Thêm prompt định dạng chặt chẽ (hoặc Structured Outputs) để model chỉ trả về đúng mảng JSON số thực: `[8.5, 2.0, 5.0]`.
3. **Mở rộng prefetch limit và hỗ trợ Payload Filter trong `Qdrant_index.py`:**
   * Cho phép truyền `limit` động từ config (`dense_top_k`, `sparse_top_k`).
   * Bổ sung `query_filter` cho Qdrant khi người dùng truyền `video_id` thay vì lọc thủ công trên RAM sau khi search.

### Giai đoạn 2: Cầu nối thị giác (Visual Retrieval) - Giải quyết dứt điểm vấn đề "mù ảnh"
Chọn 1 trong 2 hướng kiến trúc dưới đây:

* **Phương án A (Tối ưu tốc độ & không tốn VLM nặng - KHUYÊN DÙNG): Dual-Embedding trong Qdrant**
  * Tận dụng chính **SigLIP 2**: Trong Phase 2, chạy SigLIP image encoder trích xuất vector cho các keyframes của segment (tính mean-pooling thành 1 vector đại diện cho segment, dim = 1152).
  * Trong Qdrant, tạo Multi-Vector Collection gồm:
    * `dense_text`: BGE-M3 (1024)
    * `dense_visual`: SigLIP 2 (1152)
    * `sparse_text`: BGE-M3 Sparse
  * Khi search: Nhúng câu query qua cả BGE-M3 và SigLIP Text Encoder $\rightarrow$ Bắn 3 prefetch vào Qdrant và RRF gộp lại.
  * *Ưu điểm:* Cực nhanh (SigLIP chỉ là ViT encoder, nhanh hơn sinh text bằng VLM gấp hàng trăm lần), tìm kiếm hình ảnh trực tiếp (Zero-shot Cross-modal Retrieval).

* **Phương án B: Phục hồi Visual Captioning có kiểm soát (Rich Text Caption)**
  * Nếu muốn giữ cấu trúc text-only trong Qdrant: Phải sinh caption mô tả thị giác thực sự bằng một model VLM nhỏ hoặc chạy batch VLM (mô tả ngắn gọn: hành động, đối tượng, màu sắc trong 1-2 câu).
  * Không dùng mẹo sao chép OCR thành visual caption vì nó làm mất hoàn toàn nội dung hình ảnh.

### Giai đoạn 3: Tối ưu Reranker & Đa ngôn ngữ
1. Khi gọi `--rerank`, truyền ít nhất 1-2 ảnh tiêu biểu của segment cho Qwen2.5-VL (`image_paths=[...]`) để mô hình thực sự "nhìn" video khi chấm điểm.
2. Nếu câu truy vấn là tiếng Việt, có thể bổ sung bước dịch nhẹ query sang tiếng Anh trước khi đưa vào text-tower của SigLIP để tăng độ chính xác matching hình ảnh.

---

*Báo cáo được khởi tạo tự động phục vụ việc xem xét và ra quyết định kiến trúc trước khi triển khai sửa đổi mã nguồn.*

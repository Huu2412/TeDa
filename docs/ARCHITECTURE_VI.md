# TeDA: kiến trúc mã nguồn (bước 0 và bước 1)

Dự án: ViP-VL (encoder ChunkFormer) tinh chỉnh bằng CTC trên ViMD, kèm nhánh accent 3 vùng và loss căn chỉnh
với PhoBERT chỉ dùng lúc huấn luyện (suy luận chỉ cần âm thanh). Tài liệu này giải thích vai trò từng thư mục và file.

## 1. Luồng dữ liệu

```text
Chuẩn bị (chạy một lần)
  ViMD parquet ──prepare_data──▶ wav 16 kHz mono + manifest_raw.tsv
  manifest_raw ──scan_text────▶ manifest.tsv (text_clean, text_norm, region_id) + báo cáo kiểm tra
  manifest.tsv ──build_tokenizer▶ BPE-1024 (id 0 = blank, id 1 = unk), chỉ học từ tập train

Mỗi bước huấn luyện
  wav ─▶ fbank 80 chiều ─▶ [CMVN + ViP-VL 12 khối, một phần bị đóng băng] ─▶ CTC head ─▶ loss CTC
                                  (dynamic chunk ngẫu nhiên khi train)

Đánh giá:  encoder (full-context) ─▶ CTC greedy ─▶ WER/CER theo corpus, theo vùng, theo nhóm câu, khoảng tin cậy bootstrap
```

## 2. Thư mục gốc

| Đường dẫn | Vai trò |
|---|---|
| `configs/` | Mọi thiết lập. Một thí nghiệm = một file YAML nhỏ chỉ ghi phần khác `base.yaml` |
| `src/teda/` | Mã nguồn chính (package Python) |
| `third_party/chunkformer/` | Mã encoder ChunkFormer/ViP-VL sao chép nguyên bản (CC BY 4.0, commit 24b1258) |
| `scripts/` | Các lệnh chạy, mỗi file làm đúng một việc |
| `tests/` | Kiểm thử tự động, chạy được trên CPU |
| `notebooks/` | Notebook Kaggle chạy toàn bộ quy trình theo thứ tự |
| `artifacts/tokenizer/` | Tokenizer BPE đã học (nhỏ, nên commit); tạo bởi `build_tokenizer.py` |
| `outputs/<tên thí nghiệm>/` | Kết quả mỗi lần chạy: `last.pt`, `best.pt`, `metrics.jsonl`, `train.log`, `results_*.json` (không commit) |

## 3. `configs/`

| File | Vai trò |
|---|---|
| `base.yaml` | Giá trị mặc định: dữ liệu, đặc trưng, encoder và phần đóng băng, tối ưu, đánh giá |
| `exp/e00_baseline.yaml` | E00: ViP-VL + CTC. Các thí nghiệm sau (accent, alignment...) thêm file cùng thư mục |
| `sweep/freeze_search.yaml` | Danh sách cấu hình đóng băng cần thử để tìm K tối ưu |
| `vipvl_encoder.yaml` | Kiến trúc ViP-VL base dự phòng, chỉ dùng khi bundle trên Hugging Face không có `config.yaml` |
| `text_norm.yaml` | Cách đọc số (linh/lẻ, nghìn/ngàn...), kiểu đặt dấu thanh |
| `abbreviations.tsv` | Bảng viết tắt, hiện để trống (viết tắt giữ dạng chữ thường) |

## 4. `src/teda/`

| File | Vai trò |
|---|---|
| `__init__.py` | Đưa `third_party/chunkformer` vào `sys.path` để `import chunkformer` hoạt động |
| **`data/`** | **Từ dữ liệu thô đến batch** |
| `audio.py` | Giải mã, lấy kênh 0, resample 16 kHz bằng torchaudio (giống pipeline ViP-VL) |
| `prepare.py` | Chuyển từng shard parquet của ViMD, có resume, ghi `manifest_raw.tsv` |
| `text_norm.py` | Chuẩn hóa văn bản: chữ số, ngày, giờ, đơn vị thành dạng đọc; bản sạch cho PhoBERT |
| `label_maps.py` | Nhãn accent: Bắc / Trung / Nam |
| `groups.py` | Gắn nhãn nhóm câu (chữ số, viết tắt, từ mượn) để báo cáo WER riêng |
| `tokenizer.py` | BPE-1024 cho CTC (SentencePiece) |
| `features.py` | Waveform thành fbank 80 chiều, SpecAugment, speed perturb |
| `vimd_dataset.py` | Đọc `manifest.tsv`, trả fbank + token + nhãn vùng; chỉ tập train bị lọc câu quá dài hoặc CTC không khả thi |
| `sampler.py` | Gom batch theo tổng số khung (có đệm), chia đều cho từng GPU, hỗ trợ tiếp tục giữa epoch |
| `collate.py` | Đệm batch và tạo độ dài |
| **`models/`** | **Mạng nơ-ron** |
| `encoder.py` | Dựng encoder ChunkFormer từ bundle ViP-VL, nạp trọng số (báo cáo số tensor khớp, từ chối nếu quá thấp) |
| `teda_model.py` | Lắp encoder + đóng băng + CTC head; chỗ để thêm accent và alignment sau này |
| `heads/ctc_head.py` | Lớp tuyến tính + loss CTC tính ở fp32 |
| **`engine/`** | **Huấn luyện và đánh giá** |
| `trainer.py` | Vòng huấn luyện DDP, fp16, gradient accumulation, lưu/tiếp tục, giới hạn thời gian phiên Kaggle |
| `evaluator.py` | WER/CER mức corpus, theo vùng, theo nhóm, bootstrap CI, so sánh cặp giữa hai thí nghiệm |
| `decoding.py` | CTC greedy |
| **`utils/`** | **Tiện ích** |
| `config.py` | Gộp `base.yaml` + thí nghiệm + `--set`; key gõ sai bị báo lỗi |
| `freeze.py` | Đóng băng `embed`, các khối chỉ định, `after_norm`; giữ phần đóng băng ở chế độ eval nhưng encoder vẫn ở train để chọn chunk ngẫu nhiên |
| `dist.py` | Khởi tạo DDP (nccl, hoặc gloo trên CPU), gom và phát đối tượng giữa các rank |
| `ckpt.py` | Lưu nguyên tử, đọc/ghi JSON |
| `logging.py`, `seed.py`, `manifest.py` | Log (chỉ rank 0) + `metrics.jsonl`, seed theo rank, đọc/ghi TSV |

## 5. `scripts/`

| File | Việc làm |
|---|---|
| `prepare_data.py` | Bước 0a: ViMD sang wav 16 kHz mono (resume được) |
| `scan_text.py` | Bước 0b: chuẩn hóa văn bản, tạo `manifest.tsv` và báo cáo cần duyệt |
| `build_tokenizer.py` | Bước 1a: học BPE-1024 từ tập train |
| `check_encoder.py` | Chạy **đầu tiên** trên Kaggle: kiểm tra bundle ViP-VL thật nạp đúng vào encoder |
| `probe_memory.py` | Tìm `max_frames_per_batch` an toàn cho 1 GPU T4 |
| `train.py` | Huấn luyện một thí nghiệm (`torchrun --nproc_per_node=2 ...`); chạy lại cùng lệnh để tiếp tục |
| `sweep.py` | Chạy lần lượt các cấu hình đóng băng, ghi `summary.tsv` |
| `evaluate.py` | Đánh giá checkpoint; tập test chỉ chạy khi có `--final` |

## 6. Các quyết định thiết kế chính

- **Một thí nghiệm là một file YAML**, nên bảng E00 đến E04 trong báo cáo khớp từng file; `--set` đổi giá trị lẻ.
- **Đóng băng theo chỉ số khối** (`encoder.freeze_layers`, 0 đến 11) và `freeze_embed`: tìm K bằng sweep trên E00 rồi cố định cho các thí nghiệm sau.
- **Dynamic chunk** theo recipe ChunkFormer khi huấn luyện; đánh giá mặc định full-context, có thể thử chunk cố định (`--chunk 64,128,128`).
- **Chọn siêu tham số chỉ trên valid**; test chỉ dùng cho báo cáo cuối.
- **Không bỏ câu nào ở valid/test.** Ở train chỉ bỏ câu quá dài hoặc có nhiều token hơn số khung (CTC không khả thi), và ghi log số câu bị bỏ.
- **WER** là tỷ lệ lỗi âm tiết mức corpus (tổng lỗi chia tổng số từ tham chiếu), báo cáo kèm khoảng tin cậy bootstrap.

## 7. Đã kiểm chứng và chưa kiểm chứng

Đã chạy trên CPU: toàn bộ test, gồm DDP 2 tiến trình (gloo), tiếp tục sau khi bị ngắt giữa epoch, `evaluate.py`, `sweep.py`.
Cấu hình encoder dự phòng cho 78.687.744 tham số, tương đương 314,75 MB ở fp32, khớp với kích thước 315 MB của checkpoint.

**Chưa kiểm chứng (cần GPU):** fp16 autocast với GradScaler, NCCL, và nạp trọng số ViP-VL thật. Hãy chạy
`check_encoder.py` và `probe_memory.py` trên Kaggle trước khi huấn luyện.

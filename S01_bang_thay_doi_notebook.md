# Bảng thay đổi notebook `P09_Autoencoder_Claude_fixed.ipynb` so với bản gốc (S01)

> **BẢN NHÁP.** Cột "Vì sao" và "Ai viết" của các dòng em tự thay đổi để `[em điền]`. Em phải điền đúng sự thật, và ghi rõ nếu có dùng AI.
> Bản gốc để so sánh: `P09_Autoencoder_Claude_filegoc.ipynb` (64 ô). Bản hiện tại: 83 ô, **27 ô khác bản gốc** (số ô đếm từ 0 trong bản hiện tại).
> Bảng này lập bằng `diff_notebooks.py` rồi đối chiếu từng ô; chạy lại: `python diff_notebooks.py P09_Autoencoder_Claude_filegoc.ipynb P09_Autoencoder_Claude_fixed.ipynb`.

## A. Thay đổi ảnh hưởng số liệu hoặc cách đo

| # | Ô | Thay đổi | Bản gốc | Bản hiện tại | Vì sao | Ai viết | Ảnh hưởng đến số liệu |
|---|---:|---|---|---|---|---|---|
| A1 | 12 | `EVAL.target_err` | 200 | 100 | Colab hết hạn mức GPU; giảm ngân sách Monte-Carlo để chạy kịp | Em đưa notebook cho Claude (AI của Anthropic) sửa | Có: mỗi điểm BLER OFDM có ít lỗi khối hơn, nhiễu hơn (sai số ≈ 1/√số lỗi, lớn hơn khoảng 1,4 lần). |
| A2 | 12 | `EVAL.max_cw` | 100 000 | 30 000 | Colab hết hạn mức GPU; giảm ngân sách Monte-Carlo để chạy kịp | Em đưa notebook cho Claude (AI của Anthropic) sửa | Có, cho các cấu hình không phải MIMO. |
| A3 | 12 | `EVAL.gentle_max_cw` | 60 000 | 20 000 | Colab hết hạn mức GPU; giảm ngân sách Monte-Carlo để chạy kịp | Em đưa notebook cho Claude (AI của Anthropic) sửa | **Có, và thầy chưa nêu.** 2x2 dùng nhánh "gentle", nên số codeword tối đa mỗi điểm giảm 3 lần; `max_cw` không áp dụng cho 2x2. |
| A4 | 12 | `AWGN_CFG.target_err` | 1000 | 300 | Colab hết hạn mức GPU; giảm ngân sách Monte-Carlo để chạy kịp | Em đưa notebook cho Claude (AI của Anthropic) sửa | Có: sai số AWGN lớn hơn khoảng 1,8 lần. Thầy xác nhận 5/5 hệ AWGN vẫn lệch ≤ 0,013 dB. |
| A5 | 12 | `AWGN_CFG.max_cw` | 128 000 | 50 000 | Colab hết hạn mức GPU; giảm ngân sách Monte-Carlo để chạy kịp | Em đưa notebook cho Claude (AI của Anthropic) sửa | Có, như A4. |
| A6 | 8 | `train_conventional` bản vá: bỏ bước NaN/Inf, cắt gradient ở norm 100 | không có | có ở ô 8 | Chống NaN/Inf khi huấn luyện (suy ra từ mã; em chưa xác nhận lại lý do) | Theo trí nhớ của em, em đưa cho Claude sửa trong một cuộc trò chuyện khác (em chưa xác nhận lại) | **Có thể không có hiệu lực.** Ô 23 định nghĩa lại `train_conventional` (bản gốc, không có bản vá) và chạy sau ô 8, nên khi chạy từ trên xuống thì ô 23 thắng. Ô 13.2 ghi kết quả kiểm tra vào mục 8 của báo cáo. |
| A7 | 38 | `channel_statistics`: thêm `assert` NaN/Inf, `eigh` chạy trên CPU | `torch.linalg.eigh(Rf)` trên thiết bị đang dùng, không `assert` | `eigh(Rf.cpu())`, có `assert` | Báo lỗi sớm khi có NaN/Inf và tính `eigh` trên CPU (suy ra từ mã; em chưa xác nhận lại lý do) | Theo trí nhớ của em, em đưa cho Claude sửa trong một cuộc trò chuyện khác (em chưa xác nhận lại) | Chỉ đổi nơi tính; dự kiến khác ở mức sai số làm tròn. [em điền: đã đo hay chưa] |
| A8 | 32 | `LRLMMSEEstimator.freeze` bị thay bằng `_freeze_cpu` (`eigh` trên CPU, complex128, có `assert`) | `eigh` trên thiết bị đang dùng (ô 31) | trên CPU | Báo lỗi sớm khi có NaN/Inf và tính `eigh` trên CPU (suy ra từ mã; em chưa xác nhận lại lý do) | Theo trí nhớ của em, em đưa cho Claude sửa trong một cuộc trò chuyện khác (em chưa xác nhận lại) | Như A7. |
| A9 | 72 | Bảng 12.4 (AWGN) và nhận xét: cột BLER 1e-3, 1e-4 đổi số (ví dụ QAM 1e-3: 6,89 → 6,94; 1e-4: 7,37 → "> 7,5, không đo được"); nhận xét đổi ("huấn luyện 11 giây" → "khoảng 0,5 phút") | số bản gốc | số đã sửa | Cập nhật bảng và nhận xét theo lần chạy đã giảm ngân sách (suy ra từ nội dung; em chưa xác nhận lại lý do) | Theo trí nhớ của em, em đưa cho Claude sửa trong một cuộc trò chuyện khác (em chưa xác nhận lại) | Cột 1e-2 (mốc so khớp 6,49 / 6,09 / 6,06 / 6,07 / 6,22) **không đổi**. Mục 13 chỉ dùng cột 1e-2. |

## B. Thay đổi môi trường, Colab, ghi log (không đổi thuật toán)

| Ô | Bản gốc | Thay đổi | Vì sao | Ai viết | Ảnh hưởng đến số liệu |
|---:|---|---|---|---|---|
| 3 | không có | Bản đầu: `!pip install sionna==2.0.1`. Bản hiện tại: ô kiểm tra Python 3.11 và Sionna 2.0.1 | Khớp `env.yml` | Bản đầu: [em điền]. Bản hiện tại: Claude (AI) | Không |
| 4 | không có | `P09C_SMOKE`: bản đầu ghi đè `'0'`, bản hiện tại dùng `setdefault` | Cho phép bật SMOKE từ ngoài | Bản đầu: [em điền]. Bản hiện tại: Claude (AI) | Không |
| 5 | không có | Bản đầu: xóa thư mục `Claude outputs` cục bộ và mount Google Drive ở ô riêng. Bản hiện tại: gộp thành một ô xóa đúng thư mục trên Drive, ghi `clean_run.json` | Bản đầu xóa nhầm thư mục nên cache cũ trên Drive vẫn bị nạp | Bản đầu: [em điền]. Bản hiện tại: Claude (AI) | Không (đảm bảo chạy sạch) |
| 6 | không có | In thông tin GPU; bản hiện tại không lỗi khi máy không có GPU | Máy CPU bị lỗi | Bản đầu: [em điền]. Bản hiện tại: Claude (AI) | Không |
| 7 | không có | `TRAIN_BATCH = 32` | Trùng với `TRAIN_BATCH = 32` ở ô 12 gốc (cùng giá trị) | [em điền] | Không |
| 27 | không có | Bảng so khớp AWGN với P09 gốc và bảng thời gian chạy (ghi `repro_tables.md`) | Đối chiếu P09 | [em điền] | Không (chỉ tính bảng) |
| 9 | ô nhập thư viện | Thêm ghi bằng chứng xóa cache vào `run_log.txt` | Bằng chứng chạy sạch | Claude (AI) | Không |

## C. Do Claude (AI) viết sau nhận xét lần 1 và lần 2 của thầy

Em đã dùng Claude (trợ lý AI của Anthropic) và em đã yêu cầu các thay đổi này. **Không ô nào đổi thuật toán huấn luyện hay cách đo BLER.**

| Ô | Nội dung | Vì sao | Ảnh hưởng đến số liệu |
|---:|---|---|---|
| 0 | Ghi chú: hộp "Kết quả chính" là số tham chiếu | Tránh nhầm số tham chiếu với số lần chạy này | Không |
| 2 | Viết lại mục 1: một lệnh dựng môi trường, quy trình chạy sạch | Yêu cầu của thầy | Không |
| 10 | Kiểm tra đủ 3 file P09 gốc trước khi chạy | Báo lỗi sớm trên máy sạch | Không |
| 22 | Mục 3.1b: vì sao BCE là loss (nối BMD ↔ BCE) | Thầy sẽ hỏi ở buổi kiểm tra miệng | Không |
| 59, 60 | Benchmark không gọi lệnh CUDA khi máy không có GPU | Lỗi `Torch not compiled with CUDA enabled` trên máy CPU | Chỉ các số đo GPU (ghi là "—" trên máy CPU); trên GPU không đổi |
| 67 | Ghi chú nguồn số liệu mục 12 (nêu rõ 12.4 đã sửa) | Số mục 12 là số của lần chạy tham chiếu | Không |
| 75, 76 | Mục 13: tự sinh `repro_report_S01.md` | Yêu cầu báo cáo tái hiện | Không |
| 77, 78 | 13.1: chạy lại unit test mục 4.4 và ghi kết quả | Yêu cầu của kế hoạch S01 | Không |
| 79, 80 | 13.2: ghi giao thức Monte-Carlo và phiên bản mã đang chạy | Thầy yêu cầu nêu thay đổi ảnh hưởng số liệu | Không |
| 81, 82 | 13.3: chạy lại `perfect` và B0 ở 2x2 CDL-C với `EVAL` gốc | Thầy yêu cầu giải thích độ lệch 2x2 | Không (chỉ chẩn đoán) |

## D. Kết Luận

Số tái hiện trong `repro_report_S01.md` đến từ giao thức đã **giảm ngân sách Monte-Carlo** (A1 đến A5). Em có dùng Claude (AI) để viết các ô ở mục C và để viết lại các ô 3 đến 6 ở mục B. Các thay đổi A1 đến A5 do em đưa notebook cho Claude sửa vì Colab hết hạn mức GPU. Về phần A6 đến A9, chỉ A6 (bản vá huấn luyện) có thể đổi số liệu nhưng có thể không có hiệu lực vì ô 23 ghi đè (ô 13.2 sẽ xác nhận); A7–A8 chỉ đổi nơi tính eigh sang CPU (dự kiến chỉ khác ở mức sai số làm tròn, chưa đo); A9 chỉ sửa văn bản bảng 12.4, cột 1e-2 không đổi.

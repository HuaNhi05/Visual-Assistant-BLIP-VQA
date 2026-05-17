# Nghiên cứu và Triển khai Trợ lý Thị giác Thông minh dựa trên Kiến trúc BLIP-Large

Dự án tiểu luận kết thúc học phần – Nghiên cứu tối ưu hóa mô hình ngôn ngữ - thị giác quy mô nhỏ (Small Language Model - SLM) ứng dụng hỗ trợ cộng đồng người khiếm thị trong tác vụ Hỏi - Đáp thị giác (Visual Question Answering - VQA).

---

## Tập dữ liệu thực nghiệm (Dataset)
* **Link bộ dữ liệu VQAv2:** [Hugging Face Datasets - VQAv2](https://huggingface.co/datasets/merve/vqav2-small)

---

## Tính năng cốt lõi của hệ thống
* **Full-parameter Fine-tuning:** Tinh chỉnh toàn phần hệ thống trọng số của mô hình nền tảng BLIP-Large bằng thuật toán tối ưu hóa `AdamW` để thiết lập đường cơ sở chuẩn (Baseline Score).
* **Greedy Decoding Strategy:** Kích hoạt cơ chế giải mã tham lam (`num_beams=1`) trong quá trình suy luận, tối ưu hóa tốc độ phát sinh từ ngữ thời gian thực và bám sát các câu trả lời ngắn súc tích.
* **Exact Match Accuracy:** Hệ thống áp dụng cơ chế đánh giá chuỗi văn bản nghiêm ngặt. Văn bản đầu ra được chuẩn hóa qua biểu thức chính quy (Regex Normalization) để xóa bỏ mạo từ (*a, an, the*), dấu câu và đồng bộ ký tự viết thường trước khi so khớp tuyệt đối.
* **Web Assistant Interface:** Giao diện Web trực quan hỗ trợ người dùng tải lên hình ảnh thực tế và nhận phản hồi Hỏi - Đáp tức thì từ mô hình đã tinh chỉnh.

---

## Cấu trúc kho mã nguồn (Repository)

```text
Visual-Assistant-BLIP-VQA/
│
├── README.md                      # Tài liệu hướng dẫn hệ thống 
├── code.py                        # Mã train mô hình
│
└── working/
    ├── app.py                     # Mã nguồn chạy giao diện Web Demo (Local)
    │
    └── vizwiz_blip_results/       # Thư mục lưu trữ kết quả thực nghiệm
        ├── accuracy.txt           # Nhật ký ghi điểm số Exact Match Accuracy đạt được
        ├── result.json            # Tệp JSON lưu trữ 30 mẫu câu hỏi - câu trả lời thực tế
        │
        └── processor/             # Các tệp tin cấu hình bộ mã hóa ngôn ngữ (Tokenizer)
            ├── processor_config.json
            ├── tokenizer_config.json
            └── tokenizer.json

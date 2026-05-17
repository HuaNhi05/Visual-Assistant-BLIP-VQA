import os
import torch
from PIL import Image
import streamlit as st
from transformers import BlipForQuestionAnswering, BlipProcessor

# ========================================
# 1. CẤU HÌNH TRANG & CÀI ĐẶT GIAO DIỆN (CSS)
# ========================================
st.set_page_config(page_title="Visual Assistant", page_icon="👁️", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 50%, #a855f7 100%);
        color: white;
    }
    .main-title {
        color: white;
        text-align: center;
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: -1.5rem;
        margin-bottom: 0.1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .sub-title {
        color: #e0e7ff;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }
    [data-testid="stVerticalBlock"] {
        background-color: transparent !important;
        box-shadow: none !important;
        gap: 0.5rem !important;
    }
    .column-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 2px dashed rgba(255, 255, 255, 0.4) !important;
        border-radius: 12px !important;
        padding: 4px !important;
    }
    [data-testid="stImage"] img {
        max-height: 250px !important;
        width: auto !important;
        margin: 0 auto;
        border-radius: 10px;
    }
    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1f2937 !important;
        border-radius: 10px !important;
        border: 2px solid rgba(255, 255, 255, 0.5) !important;
        padding: 0.4rem 0.8rem !important;
        font-size: 0.95rem;
    }
    .stTextInput input::placeholder {
        color: #6b7280 !important;
        opacity: 1 !important;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #a855f7 0%, #818cf8 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 10px !important;
        width: 100%;
        font-size: 1rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .response-box {
        background-color: rgba(255, 255, 255, 0.95);
        border-left: 6px solid #a855f7;
        padding: 1.2rem;
        border-radius: 14px;
        font-size: 1.05rem;
        color: #1f2937;
        line-height: 1.5;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        min-height: 150px;
    }
    </style>
    """, unsafe_allow_html=True)

# ========================================
# 2. KHỞI TẠO STATE & HÀM CALLBACK FIX LỖI
# ========================================
if "saved_question" not in st.session_state:
    st.session_state.saved_question = ""

def on_image_change():
    # Xóa sạch bộ nhớ biến logic
    st.session_state.saved_question = ""
    # Ép buộc xóa sạch giá trị hiện tại của ô nhập liệu text_input trong bộ nhớ widget
    if "text_question_input" in st.session_state:
        st.session_state.text_question_input = ""

# ========================================
# 3. NẠP MÔ HÌNH BLIP TỪ LOCAL
# ========================================
@st.cache_resource
def load_local_model():
    BASE_DIR = "./vizwiz_blip_results" 
    MODEL_PATH = os.path.join(BASE_DIR, "model")
    PROCESSOR_PATH = os.path.join(BASE_DIR, "processor")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained(PROCESSOR_PATH)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    return model, processor, device

try:
    model, processor, device = load_local_model()
except Exception as e:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to(device)
    model.eval()

# ========================================
# 4. THIẾT KẾ BỐ CỤC GIAO DIỆN COMPACT
# ========================================

st.markdown('<div class="main-title">👁️ Visual Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Trợ lý AI thông minh cho người khiếm thị - Upload ảnh và hỏi bất cứ điều gì!</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown('<div class="column-header">🔵 Nhập hình ảnh</div>', unsafe_allow_html=True)
    
    # Kéo thả ảnh tích hợp hàm callback đã được sửa lỗi xóa bộ nhớ đệm
    uploaded_file = st.file_uploader(
        "Kéo thả hoặc click để chọn ảnh", 
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        on_change=on_image_change
    )
    
    if uploaded_file is not None:
        image_display = Image.open(uploaded_file)
        st.image(image_display, use_container_width=False)
    
    # Ô nhập liệu đồng bộ hai chiều mượt mà
    question = st.text_input(
        "Hỏi về hình ảnh...",
        placeholder="VD: How many chairs are there? What color is the car?",
        label_visibility="collapsed",
        key="text_question_input"
    )
    
    # Lưu giá trị hiện tại vào biến trạng thái
    st.session_state.saved_question = question

    submit_button = st.button("🎤 Phân tích & Trả lời")

with col2:
    st.markdown('<div class="column-header">🧩 Kết quả trả lời</div>', unsafe_allow_html=True)
    
    # Chỉ xử lý phân tích nếu có ảnh được tải lên
    if uploaded_file is not None:
        if submit_button and question.strip() != "":
            with st.spinner("🤖 AI đang phân tích..."):
                raw_image = Image.open(uploaded_file).convert("RGB")
                inputs = processor(images=raw_image, text=str(question), return_tensors="pt").to(device)
                
                with torch.no_grad():
                    out = model.generate(**inputs, max_length=15, num_beams=3)
                answer = processor.decode(out[0], skip_special_tokens=True)
                
            st.markdown(f"""
                <div class="response-box">
                    <span style="color: #6b7280; font-size: 0.85rem; font-weight: bold;">CÂU HỎI:</span>
                    <strong style="color: #1f2937; font-size: 1.1rem; display: block; margin-top: 2px;">{question}</strong>
                    <hr style="border: 0; border-top: 1px solid #e5e7eb; margin: 0.6rem 0;">
                    <span style="color: #a855f7; font-weight: bold; font-size: 0.85rem;">🤖 AI PHẢN HỒI:</span><br>
                    <span style="font-size: 1.5rem; font-weight: 800; color: #10b981; letter-spacing: 0.5px;">{answer.upper()}</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Khi có ảnh nhưng chưa bấm nút hoặc gõ câu hỏi sai đang sửa lại
            st.markdown("""
                <div class="response-box" style="color: #4b5563; border-left: 6px solid #a855f7;">
                    <span style="font-size: 1.1rem; font-weight: 600; color: #374151;">Đã nhận hình ảnh! 📸</span><br>
                    Mời bạn nhập câu hỏi bằng tiếng Anh vào ô bên dưới và nhấn <strong>Phân tích & Trả lời</strong>.
                </div>
            """, unsafe_allow_html=True)
    else:
        # Khi hoàn toàn trống ảnh (mới mở trang, hoặc vừa bấm nút Xóa ảnh / Đổi ảnh)
        st.markdown("""
            <div class="response-box" style="color: #4b5563; border-left: 6px solid #cbd5e1;">
                <span style="font-size: 1.1rem; font-weight: 600; color: #374151;">Chào bạn! 👋</span><br>
                Hãy tải một hình ảnh lên hệ thống và nhập câu hỏi tương tác.<br>
                Hệ thống lõi AI sẽ lập tức đưa ra câu trả lời chính xác nhất.
            </div>
        """, unsafe_allow_html=True)
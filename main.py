import os
import uuid
import shutil
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from rembg import remove

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Bật CORS để cho phép Frontend kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tạo các thư mục lưu tạm dữ liệu
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cho phép truy cập file 3D tĩnh qua URL
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Khởi tạo AI Model TripoSR (Tự động tải weights nếu chưa có)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Đang chạy mô hình AI trên thiết bị: {device}")

try:
    from tsr.system import TSR
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.json", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to(device)
except Exception as e:
    print(f"Cảnh báo: Chưa thể load TripoSR ({e}). Vui lòng kiểm tra kết nối internet.")
    model = None

@app.post("/api/generate-3d")
async def generate_3d(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file ảnh định dạng PNG, JPG, WEBP.")

    session_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
    rembg_path = os.path.join(UPLOAD_DIR, f"nobg_{session_id}.png")
    output_glb_path = os.path.join(OUTPUT_DIR, f"{session_id}.glb")

    # 1. Lưu file ảnh tải lên
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Xóa nền ảnh (Background Removal)
    input_img = Image.open(input_path)
    nobg_img = remove(input_img)
    nobg_img.save(rembg_path)

    # 3. Chuyển ảnh thành 3D bằng TripoSR
    if model is not None:
        with torch.no_grad():
            scene_codes = model([nobg_img], device=device)
            meshes = model.extract_mesh(scene_codes, resolution=256)
            meshes[0].export(output_glb_path)
    else:
        raise HTTPException(status_code=500, detail="Model TripoSR chưa sẵn sàng.")

    return {
        "status": "success",
        "glb_url": f"/outputs/{session_id}.glb"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

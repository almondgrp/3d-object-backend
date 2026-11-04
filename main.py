import os
import uuid
import shutil
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from rembg import remove

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

model = None
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_model():
    global model
    if model is None:
        print(f"Đang tải mô hình AI TripoSR vào thiết bị: {device}...")
        from tsr.system import TSR
        model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.json", weight_name="model.ckpt")
        model.renderer.set_chunk_size(4096)
        model.to(device)
    return model

@app.get("/")
def home():
    return {"status": "Backend is running!"}

@app.post("/api/generate-3d")
async def generate_3d(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file ảnh định dạng PNG, JPG, WEBP.")

    session_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
    rembg_path = os.path.join(UPLOAD_DIR, f"nobg_{session_id}.png")
    output_glb_path = os.path.join(OUTPUT_DIR, f"{session_id}.glb")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    input_img = Image.open(input_path)
    nobg_img = remove(input_img)
    nobg_img.save(rembg_path)

    ai_model = get_model()
    with torch.no_grad():
        scene_codes = ai_model([nobg_img], device=device)
        meshes = ai_model.extract_mesh(scene_codes, resolution=128)
        meshes[0].export(output_glb_path)

    return {
        "status": "success",
        "glb_url": f"/outputs/{session_id}.glb"
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

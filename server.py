import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import torch.nn.functional as F
import base64
import io
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from Models import SimpleMNISTCNN

app = FastAPI()

# --- Mount Static Files ---
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Model Setup ---
def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleMNISTCNN().to(device)
    checkpoint_path = './Exps/Example_Training_1/model_best.pt'
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        model.eval() 
        print("Model loaded successfully!")
        return model, device
    except FileNotFoundError:
        print(f"Could not find model weights at {checkpoint_path}.")
        exit()

model, device = load_model()

# --- API Endpoint ---
class ImageData(BaseModel):
    image: str

@app.post("/predict")
async def predict(data: ImageData):
    encoded_data = data.image.split(',')[1]
    img_bytes = base64.b64decode(encoded_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('L') 
    img = ImageOps.invert(img)
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)), 
        transforms.ToTensor(),       
        transforms.Normalize((0.1307,), (0.3081,)) 
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output[0], dim=0) 
        
    probs, classes = torch.topk(probabilities, 10)
    results = [{"class": str(classes[i].item()), "prob": round(probs[i].item() * 100, 2)} for i in range(10)]
    
    return {"predictions": results}

# --- Serve the UI ---
@app.get("/")
async def get_ui():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
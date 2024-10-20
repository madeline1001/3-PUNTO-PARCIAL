from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.models as models  
import torchvision.transforms as transforms
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
from PIL import Image

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

device = torch.device("cpu")  

model = models.resnet18()  
model.fc = torch.nn.Linear(512, 100)  

model.load_state_dict(torch.load('cifar100_best_model.pkl', map_location=torch.device('cpu')))
model = model.to(device)
model.eval()  

@app.post("/predict-image/")
async def upload_image(file: UploadFile = File(...)):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  
    ])
    
    class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bird',
                   'boat', 'book', 'bottle', 'bowling_ball', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat',
                   'cattle', 'chair', 'clock', 'cloud', 'computer', 'couch', 'cow', 'crab', 'crocodile', 'cup',
                   'dinosaur', 'dog', 'donut', 'drum', 'elephant', 'flatfish', 'forest', 'fox', 'frog', 'girl',
                   'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
                   'lobster', 'man', 'map', 'motorcycle', 'mouse', 'mushroom', 'otter', 'palm_tree', 'pear', 
                   'penguin', 'pickup_truck', 'pizza', 'platypus', 'pomegranate', 'porcupine', 'rabbit', 'raccoon',
                   'ray', 'road', 'rocket', 'sandwich', 'saucer', 'scorpion', 'seal', 'shark', 'sheep', 'skateboard',
                   'skull', 'snail', 'snake', 'snowman', 'sofa', 'spider', 'squirrel', 'starfish', 'strawberry',
                   'suitcase', 'sunglasses', 'table', 'tiger', 'toaster', 'tortoise', 'traffic_light', 'train',
                   'truck', 'umbrella', 'whale', 'zebra']
    
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    imagen = Image.open(file_path).convert("RGB")
    imagen_t = transform(imagen)
    imagen_t = imagen_t.unsqueeze(0)  
    imagen_t = imagen_t.to(device)
    
    with torch.no_grad():
        outputs = model(imagen_t)
        _, pred = torch.max(outputs, 1)

    file_path.unlink()

    return JSONResponse({"Clase": class_names[pred.item()]})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

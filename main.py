from fastapi import FastAPI
from fastapi import UploadFile, File
import uvicorn
from detect import read, predict

app = FastAPI()

@app.get('/index')
def hello_world(name: str):
    return f"Hello {name}"

@app.post('/api/predict')
async def predict(file: UploadFile = File(...)):
    sfile = await file.read()
    image = read(sfile)
    preds = predict(image)
    
    return preds

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')

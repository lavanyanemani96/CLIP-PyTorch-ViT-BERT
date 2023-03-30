import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
from application.components import predict, read_imagefile
from fastapi.responses import FileResponse

app = FastAPI(title='FastAPI CLIP-PyTorch (ViT and BERT)')

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.post("/predict/image")
async def predict_image_api(file: UploadFile = File(...)):

    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())

    image_name, caption = predict(image)
    image_name = '/home/lavanya/Downloads/Industry/Inkers/TechnicalRound2/CLIP-PyTorch-ViT-BERT/application/components/prediction/flickr30k_images/'+image_name

    return FileResponse(image_name)

@app.post("/predict/caption")
async def predict_caption_api(file: UploadFile = File(...)):

    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())

    image_name, caption = predict(image)

    return caption

if __name__ == "__main__":
    uvicorn.run(app, debug=True)

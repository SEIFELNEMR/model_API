# main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
from model_prediction import predict_disease

app = FastAPI()

# Ensure the temporary directory exists
# TEMP_DIR = "temp"
# os.makedirs(TEMP_DIR, exist_ok=True)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_location = os.path.join(file.filename)

    # Save the uploaded file
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Predict the disease
    result = predict_disease(file_location)

    if "error" in result:
        return JSONResponse(content=result)

    # Delete the temporary input file
    os.remove(file_location)

    # Serve the output image
    image_url = f"/images/{os.path.basename(result['image_path'])}"
    return JSONResponse(content={"prediction": result["prediction"], "image_url": image_url})


@app.get("/images/{image_name}")
async def get_image(image_name: str):
    image_path = os.path.join(image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    return JSONResponse(content={"error": "Image not found"}, status_code=404)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
if __name__ == "__main__":
    main()

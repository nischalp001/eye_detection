from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow only these origins
    allow_credentials=True,
    allow_methods=["*"],    # allow all HTTP methods
    allow_headers=["*"],    # allow all headers
)

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def find_eye_pairs(eyes, max_vertical_diff=20, max_horizontal_gap=150):
    pairs = []
    eyes = sorted(eyes, key=lambda e: e[0])  # sort by x (left to right)
    for i in range(len(eyes)):
        for j in range(i + 1, len(eyes)):
            x1, y1, w1, h1 = eyes[i]
            x2, y2, w2, h2 = eyes[j]
            vertical_diff = abs(y1 - y2)
            horizontal_gap = abs((x2 + w2 / 2) - (x1 + w1 / 2))
            if vertical_diff <= max_vertical_diff and horizontal_gap <= max_horizontal_gap:
                pairs.append((eyes[i], eyes[j]))
    return pairs

@app.post("/detect_eye_pair/")
async def detect_eye_pair(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=8)

    pairs = find_eye_pairs(eyes)

    if not pairs:
        return {"message": "No eye pairs detected", "pair": None}

    # Closest pair = largest combined area
    def pair_area(pair):
        (x1, y1, w1, h1), (x2, y2, w2, h2) = pair
        return w1 * h1 + w2 * h2

    closest_pair = max(pairs, key=pair_area)

    pair_response = [
        {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
        for (x, y, w, h) in closest_pair
    ]

    return {
        "message": "Closest eye pair detected",
        "pair": pair_response
    }

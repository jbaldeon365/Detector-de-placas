from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2, re, os, imutils

# Cargar modelos solo una vez
model = YOLO("best_3.pt")  # Ruta al modelo YOLO entrenado
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def detect_plate(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return [], None
    
    results = model(image)
    plate_texts = []

    for result in results:
        index_plates = (result.boxes.cls == 0).nonzero(as_tuple=True)[0]

        for idx in index_plates:
            conf = result.boxes.conf[idx].item()
            if conf > 0.05:
                xyxy = result.boxes.xyxy[idx].squeeze().tolist()
                x1, y1, x2, y2 = map(int, xyxy)

                y1p, y2p = max(y1 - 10, 0), y2 + 10
                x1p, x2p = max(x1 - 10, 0), x2 + 10
                plate_img = image[y1p:y2p, x1p:x2p]

                if plate_img.size == 0:
                    continue

                result_ocr = ocr.ocr(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB), cls=True)
                if not result_ocr or not result_ocr[0]:
                    continue

                texts = [line[1][0] for line in result_ocr[0]]
                raw_text = ''.join(texts)
                clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
                match = re.search(r'[A-Z]{2,3}\d{3,4}', clean_text)

                if match:
                    output_text = match.group()
                    plate_texts.append(output_text)
                else:
                    output_text = clean_text

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, output_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    image = imutils.resize(image, width=720)
    annotated_path = os.path.join("uploads", "annotated.jpg")
    cv2.imwrite(annotated_path, image)
    return plate_texts, annotated_path

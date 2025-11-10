from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import re
import os
import imutils

# --- CAMBIO ---
# Define el directorio temporal escribible en Render
TEMP_DIR = "/tmp"

# --- ADVERTENCIA ---
# Cargar modelos solo una vez.
# ¡ASEGÚRATE DE QUE 'best_3.pt' ESTÉ INCLUIDO EN TU REPOSITORIO DE GITHUB!
# Si este archivo no está, la app fallará al iniciar.
model = YOLO("best_3.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def detect_plate(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo leer la imagen en {image_path}")
        return [], None
    
    results = model(image)
    plate_texts = []

    for result in results:
        # (Toda tu lógica de detección de placas va aquí... es correcta)
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

    # --- CAMBIOS CRÍTICOS PARA RENDER ---
    
    # 1. Redimensionar la imagen anotada
    image = imutils.resize(image, width=720)
    
    # 2. Crear un nombre de archivo de salida ÚNICO
    #    (Basado en el nombre del archivo original para evitar que los usuarios se pisen)
    input_filename = os.path.basename(image_path)
    annotated_filename = f"annotated_{input_filename}"
    
    # 3. Definir la ruta de guardado en el directorio temporal /tmp
    annotated_path = os.path.join(TEMP_DIR, annotated_filename)
    
    try:
        # 4. Guardar la imagen anotada en /tmp
        cv2.imwrite(annotated_path, image)
    except Exception as e:
        print(f"Error al guardar la imagen anotada en {annotated_path}: {e}")
        return plate_texts, None # Devolver None si falla el guardado

    # 5. Devolver las placas y la RUTA COMPLETA al archivo en /tmp
    return plate_texts, annotated_path
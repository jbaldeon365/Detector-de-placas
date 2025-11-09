# detector.py
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import re
import os
import imutils

# Cargar modelos solo una vez al iniciar el servidor
model = YOLO("D:/Python/Tf_SI/best_3.pt")  # Ruta al modelo YOLO entrenado
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # OCR con corrección de inclinación

def detect_plate(image_path):
    """
    Detecta placas vehiculares en una imagen, aplica OCR y devuelve el texto reconocido
    junto con la ruta de la imagen anotada.
    """
    image = cv2.imread(image_path)
    if image is None:
        return [], None  # Si no se carga la imagen, retornamos vacío
    
    results = model(image)
    plate_texts = []

    for result in results:
        # Solo detecciones de clase 'placa' (cls == 0)
        index_plates = (result.boxes.cls == 0).nonzero(as_tuple=True)[0]

        for idx in index_plates:
            conf = result.boxes.conf[idx].item()
            if conf > 0.05:  # Confianza mínima igual al código original
                xyxy = result.boxes.xyxy[idx].squeeze().tolist()
                x1, y1 = int(xyxy[0]), int(xyxy[1])
                x2, y2 = int(xyxy[2]), int(xyxy[3])

                # Recortar imagen de la placa con menos padding (reduce texto extra)
                y1_pad, y2_pad = max(y1 - 10, 0), y2 + 10
                x1_pad, x2_pad = max(x1 - 10, 0), x2 + 10
                plate_image = image[y1_pad:y2_pad, x1_pad:x2_pad]

                if plate_image.size == 0:
                    continue

                # Ejecutar OCR
                result_ocr = ocr.predict(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
                boxes = result_ocr[0]['rec_boxes']
                texts = result_ocr[0]['rec_texts']

                # Ordenar y unir textos
                left_to_right = sorted(zip(boxes, texts), key=lambda x: min(x[0][::2]))
                raw_text = ''.join([t for _, t in left_to_right])

                # 🔹 Filtrar textos: solo letras y números
                clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

                # 🔹 Validar formato de placa (ej: ABC123 o AB1234)
                match = re.search(r'[A-Z]{2,3}\d{3,4}', clean_text)
                if match:
                    output_text = match.group()
                    plate_texts.append(output_text)
                else:
                    output_text = clean_text  # opcional, para depurar

                # Dibujar
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, output_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Redimensionar imagen final (como el código original)
    image = imutils.resize(image, width=720)

    # Guardar imagen anotada
    annotated_path = os.path.join("uploads", "annotated.jpg")
    cv2.imwrite(annotated_path, image)

    return plate_texts, annotated_path

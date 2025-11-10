from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
# ¡Asegúrate que detector.py guarde sus resultados en /tmp!
from detector import detect_plate

app = Flask(__name__, template_folder="templates", static_folder="static")

# --- CAMBIO ---
# Define el directorio temporal que SÍ es escribible en Render.
# Ya no usamos app.config["UPLOAD_FOLDER"].
TEMP_DIR = "/tmp"

# Cargar JSON local con placas de vehículos (esto está bien)
def cargar_vehiculos():
    # Asumimos que 'vehiculos.json' está en la raíz del proyecto
    with open("vehiculos.json", "r", encoding="utf-8") as f:
        return json.load(f)["vehiculos"]

@app.route("/")
def index():
    return render_template("index.html")

# --- CAMBIO ---
# Esta ruta servirá los archivos temporales (imágenes subidas y anotadas)
# desde el directorio /tmp. La he renombrado de '/uploads/' a '/temp_files/'
# para que sea más claro.
@app.route("/temp_files/<filename>")
def temp_file(filename):
    return send_from_directory(TEMP_DIR, filename)

@app.route("/api/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No se envió ninguna imagen."})

    file = request.files["image"]
    
    if file.filename == '':
        return jsonify({"success": False, "error": "Se seleccionó un archivo sin nombre."})

    filename = secure_filename(file.filename)
    
    # --- CAMBIO ---
    # Guardamos la imagen subida en el directorio /tmp
    filepath = os.path.join(TEMP_DIR, filename)
    
    try:
        file.save(filepath)
    except Exception as e:
        # Añadimos un manejo de error por si falla el guardado en /tmp
        print(f"Error guardando archivo en /tmp: {e}")
        return jsonify({"success": False, "error": "No se pudo guardar el archivo en el servidor."})

    # --- ASUNCIÓN CRÍTICA ---
    # Asumimos que 'detect_plate' guarda su imagen anotada en /tmp
    # y devuelve la RUTA COMPLETA a ese archivo (ej: '/tmp/annotated_image.jpg')
    try:
        plates, annotated_path = detect_plate(filepath)
    except Exception as e:
        # Si el detector falla (ej: error de Ultralytics), lo capturamos
        print(f"Error durante la detección: {e}")
        return jsonify({"success": False, "error": f"Error interno durante la detección: {e}"})

    if plates:
        vehiculos = cargar_vehiculos()
        resultados = []

        for p in plates:
            p_upper = p.strip().upper()
            vehiculo = next((v for v in vehiculos if v["placa"] == p_upper), None)
            if vehiculo:
                resultados.append({
                    "placa": p_upper,
                    "encontrado": True,
                    "estado": vehiculo["estado"],
                    "marca": vehiculo["marca"],
                    "modelo": vehiculo["modelo"],
                    "color": vehiculo["color"],
                    "descripcion": vehiculo["descripcion"]
                })
            else:
                resultados.append({
                    "placa": p_upper,
                    "encontrado": False,
                    "estado": "desconocido",
                    "descripcion": "No hay registro en la base local."
                })

        # --- CAMBIO ---
        # Obtenemos solo el nombre del archivo de la ruta devuelta por detect_plate
        # (Ej: de '/tmp/annotated.jpg' extraemos 'annotated.jpg')
        annotated_filename = os.path.basename(annotated_path)

        return jsonify({
            "success": True,
            "resultados": resultados,
            # Creamos la URL para que el frontend la pida a nuestra ruta /temp_files/
            "annotated": f"/temp_files/{annotated_filename}"
        })

    else:
        return jsonify({"success": False, "error": "No se detectaron placas."})

# Este bloque solo se ejecuta si corres 'python app.py' localmente.
# Gunicorn no lo usa, pero es bueno para pruebas locales.
if __name__ == "__main__":
    # Añadimos debug=True para ver errores más fácil en local
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
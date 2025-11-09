from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os, json
from detector import detect_plate

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")

# Cargar JSON local con placas de vehículos sospechosos o robados
def cargar_vehiculos():
    with open("vehiculos.json", "r", encoding="utf-8") as f:
        return json.load(f)["vehiculos"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/api/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No se envió ninguna imagen."})

    file = request.files["image"]
    filename = secure_filename(file.filename)
    filepath = os.path.join("uploads", filename)
    file.save(filepath)

    plates, annotated_path = detect_plate(filepath)

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

        return jsonify({
            "success": True,
            "resultados": resultados,
            "annotated": f"/uploads/annotated.jpg"
        })

    else:
        return jsonify({"success": False, "error": "No se detectaron placas."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

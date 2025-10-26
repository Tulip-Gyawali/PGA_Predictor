# frontend/app.py
from flask import Flask, request, render_template, redirect, url_for, flash
import requests, os
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "dev-secret"
API_URL = "http://127.0.0.1:8000"  # where you run the FastAPI backend

UPLOAD_FOLDER = "frontend/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("file")
        model_type = request.form.get("model_type", "xgb")
        if not f:
            flash("Please upload a CSV file")
            return redirect(request.url)
        filename = secure_filename(f.filename)
        local_path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(local_path)
        files = {"file": open(local_path, "rb")}
        data = {"model_type": model_type}
        resp = requests.post(f"{API_URL}/predict_file/", files=files, data=data)
        if resp.status_code == 200:
            info = resp.json()
            return render_template("index.html", result=info)
        else:
            flash(f"API error: {resp.text}")
            return redirect(request.url)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)

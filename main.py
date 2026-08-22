"""
Görsel Sahtecilik Tespit Sistemi - Faz 1
Fotoğraf yükleme + EXIF/Metadata analizi

Çalıştırmak için:
    pip install -r requirements.txt
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Sonra tarayıcıda: http://localhost:8000
"""

import io
import hashlib
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

app = FastAPI(title="Görsel Sahtecilik Tespit Sistemi")

# Farklı telefon/tarayıcılardan erişim için CORS açık bırakıyoruz.
# Prodüksiyona alırken allow_origins listesini kendi domain'inle sınırla.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Fotoğraf düzenleme yazılımlarının EXIF'te sıkça bıraktığı izler.
SUSPICIOUS_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "lightroom", "affinity photo",
    "pixelmator", "snapseed", "picsart", "facetune",
]


def extract_exif(image: Image.Image) -> dict:
    """Pillow ile EXIF etiketlerini okunabilir sözlüğe çevirir."""
    exif_data = {}
    raw_exif = image.getexif()

    if not raw_exif:
        return exif_data

    for tag_id, value in raw_exif.items():
        tag_name = TAGS.get(tag_id, tag_id)

        # GPS bilgisi ayrı bir alt yapı olduğu için özel işliyoruz
        if tag_name == "GPSInfo":
            gps_data = {}
            for gps_tag_id, gps_value in value.items():
                gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                gps_data[gps_tag_name] = gps_value
            exif_data["GPSInfo"] = gps_data
            continue

        # bytes tipindeki değerleri JSON'a uygun hale getir
        if isinstance(value, bytes):
            try:
                value = value.decode(errors="replace")
            except Exception:
                value = str(value)

        exif_data[str(tag_name)] = value if not isinstance(value, (tuple, list)) else [str(v) for v in value]

    return exif_data


def analyze_metadata(exif_data: dict) -> dict:
    """EXIF verisinden basit şüphe sinyalleri üretir."""
    flags = []
    score = 0  # 0 = temiz, yüksek = şüpheli

    software = str(exif_data.get("Software", "")).lower()
    if software:
        for keyword in SUSPICIOUS_SOFTWARE_KEYWORDS:
            if keyword in software:
                flags.append(f"Düzenleme yazılımı izi bulundu: '{exif_data.get('Software')}'")
                score += 40
                break

    if not exif_data:
        flags.append("EXIF verisi tamamen boş — ekran görüntüsü, indirilmiş görsel veya metadata temizlenmiş olabilir.")
        score += 20

    has_gps = "GPSInfo" in exif_data and bool(exif_data["GPSInfo"])
    has_datetime = "DateTime" in exif_data or "DateTimeOriginal" in exif_data

    if not has_datetime and exif_data:
        flags.append("Çekim tarihi bilgisi yok.")
        score += 10

    return {
        "suspicion_score": min(score, 100),
        "flags": flags,
        "has_gps": has_gps,
        "has_datetime": has_datetime,
        "software_tag": exif_data.get("Software"),
    }


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sadece görsel dosyası yükleyebilirsin.")

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Boş dosya yüklendi.")

    # Dosyayı kaydet (ileride copy-move / ELA analizinde kullanacağız)
    file_hash = hashlib.sha256(contents).hexdigest()[:16]
    saved_path = UPLOAD_DIR / f"{file_hash}_{file.filename}"
    saved_path.write_bytes(contents)

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()  # dosya bozuk mu kontrol et
        image = Image.open(io.BytesIO(contents))  # verify() sonrası tekrar aç
    except Exception:
        raise HTTPException(status_code=400, detail="Dosya geçerli bir görsel değil veya bozuk.")

    exif_data = extract_exif(image)
    metadata_analysis = analyze_metadata(exif_data)

    result = {
        "filename": file.filename,
        "file_hash": file_hash,
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
        "image_info": {
            "format": image.format,
            "size": image.size,
            "mode": image.mode,
        },
        "exif": exif_data,
        "metadata_analysis": metadata_analysis,
    }

    return JSONResponse(content=jsonable(result))


def jsonable(obj):
    """EXIF içindeki bazı özel tipleri (IFDRational vb.) JSON'a çevirir."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


@app.get("/", response_class=HTMLResponse)
async def home():
    """Test için basit bir upload arayüzü (HTML doğrudan koda gömülü)."""
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Görsel Sahtecilik Tespit Sistemi</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #171a21;
    --border: #2a2f3a;
    --text: #e6e8eb;
    --muted: #8b93a3;
    --accent: #ff6b35;
    --accent-dim: #ff6b3522;
    --danger: #e5484d;
    --ok: #3ecf8e;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px 16px 60px;
  }
  .wrap { max-width: 640px; margin: 0 auto; }
  h1 {
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0 0 4px;
    letter-spacing: -0.01em;
  }
  .subtitle { color: var(--muted); font-size: 0.9rem; margin-bottom: 24px; }

  .dropzone {
    border: 2px dashed var(--border);
    border-radius: 14px;
    padding: 36px 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.15s ease, background 0.15s ease;
    background: var(--panel);
  }
  .dropzone.drag { border-color: var(--accent); background: var(--accent-dim); }
  .dropzone p { margin: 8px 0 0; color: var(--muted); font-size: 0.85rem; }
  .dropzone strong { color: var(--text); font-size: 1rem; }
  input[type=file] { display: none; }

  #preview { margin-top: 16px; display: none; }
  #preview img {
    width: 100%;
    border-radius: 12px;
    border: 1px solid var(--border);
    display: block;
  }

  button.analyze {
    width: 100%;
    margin-top: 16px;
    padding: 14px;
    border: none;
    border-radius: 10px;
    background: var(--accent);
    color: #0f1115;
    font-weight: 700;
    font-size: 1rem;
    cursor: pointer;
    display: none;
  }
  button.analyze:active { transform: scale(0.98); }
  button.analyze:disabled { opacity: 0.6; }

  .report {
    margin-top: 24px;
    display: none;
  }
  .score-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    margin-bottom: 16px;
  }
  .score-num { font-size: 2.4rem; font-weight: 800; line-height: 1; }
  .score-label { color: var(--muted); font-size: 0.8rem; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.05em; }

  .section {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 12px;
  }
  .section h3 {
    margin: 0 0 10px;
    font-size: 0.85rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    font-size: 0.9rem;
    border-bottom: 1px solid var(--border);
  }
  .row:last-child { border-bottom: none; }
  .row span:first-child { color: var(--muted); }

  .flag {
    display: flex;
    gap: 8px;
    padding: 8px 0;
    font-size: 0.87rem;
    align-items: flex-start;
  }
  .flag::before { content: "⚠️"; flex-shrink: 0; }
  .flag.clean::before { content: "✅"; }
  .flag.clean { color: var(--ok); }

  #status { text-align: center; color: var(--muted); font-size: 0.85rem; margin-top: 10px; display: none; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🔍 Görsel Sahtecilik Tespit Sistemi</h1>
  <div class="subtitle">Faz 1 — Metadata / EXIF analizi</div>

  <div class="dropzone" id="dropzone">
    <strong>Fotoğraf seç ya da sürükle</strong>
    <p>JPG, PNG desteklenir</p>
    <input type="file" id="fileInput" accept="image/*">
  </div>

  <div id="preview"><img id="previewImg" alt="önizleme"></div>
  <button class="analyze" id="analyzeBtn">Analiz Et</button>
  <div id="status">Analiz ediliyor…</div>

  <div class="report" id="report">
    <div class="score-card">
      <div class="score-num" id="scoreNum">0</div>
      <div class="score-label">Şüphe Skoru / 100</div>
    </div>

    <div class="section">
      <h3>Bulgular</h3>
      <div id="flagsList"></div>
    </div>

    <div class="section">
      <h3>Dosya Bilgisi</h3>
      <div id="fileInfo"></div>
    </div>
  </div>
</div>

<script>
const API_URL = window.location.origin + "/analyze";

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const previewImg = document.getElementById("previewImg");
const analyzeBtn = document.getElementById("analyzeBtn");
const statusEl = document.getElementById("status");
const report = document.getElementById("report");

let selectedFile = null;

dropzone.addEventListener("click", () => fileInput.click());

["dragover", "dragenter"].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove("drag"); })
);
dropzone.addEventListener("drop", e => {
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", e => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    preview.style.display = "block";
    analyzeBtn.style.display = "block";
    report.style.display = "none";
  };
  reader.readAsDataURL(file);
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  statusEl.style.display = "block";
  report.style.display = "none";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch(API_URL, { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Analiz başarısız oldu.");
    }
    const data = await res.json();
    renderReport(data);
  } catch (err) {
    alert("Hata: " + err.message);
  } finally {
    analyzeBtn.disabled = false;
    statusEl.style.display = "none";
  }
});

function renderReport(data) {
  const analysis = data.metadata_analysis;

  document.getElementById("scoreNum").textContent = analysis.suspicion_score;
  document.getElementById("scoreNum").style.color =
    analysis.suspicion_score >= 50 ? "#e5484d" :
    analysis.suspicion_score >= 20 ? "#ff6b35" : "#3ecf8e";

  const flagsList = document.getElementById("flagsList");
  flagsList.innerHTML = "";
  if (analysis.flags.length === 0) {
    const d = document.createElement("div");
    d.className = "flag clean";
    d.textContent = "Metadata'da belirgin bir şüphe işareti bulunamadı.";
    flagsList.appendChild(d);
  } else {
    analysis.flags.forEach(f => {
      const d = document.createElement("div");
      d.className = "flag";
      d.textContent = f;
      flagsList.appendChild(d);
    });
  }

  const fileInfo = document.getElementById("fileInfo");
  fileInfo.innerHTML = "";
  const rows = [
    ["Dosya adı", data.filename],
    ["Boyut", data.image_info.size.join(" × ") + " px"],
    ["Format", data.image_info.format],
    ["GPS bilgisi", analysis.has_gps ? "Var" : "Yok"],
    ["Tarih bilgisi", analysis.has_datetime ? "Var" : "Yok"],
    ["Yazılım etiketi", analysis.software_tag || "—"],
  ];
  rows.forEach(([label, val]) => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<span>${label}</span><span>${val}</span>`;
    fileInfo.appendChild(row);
  });

  report.style.display = "block";
}
</script>
</body>
</html>
"""


@app.get("/health")
async def health():
    return {"status": "ok"}

"""
Görsel Sahtecilik Tespit Sistemi - Faz 1
Fotoğraf yükleme + EXIF/Metadata analizi

Çalıştırmak için:
    pip install -r requirements.txt
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Sonra tarayıcıda: http://localhost:8000
"""

import io
import base64
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, ImageChops, ImageEnhance, ImageDraw
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


def compute_ela(image: Image.Image, quality: int = 90, scale: int = 15) -> tuple[Image.Image, dict]:
    """
    Error Level Analysis: görseli belirli bir JPEG kalitesinde yeniden
    kaydedip orijinaliyle arasındaki farkı hesaplar. Dokunulmamış bölgeler
    her yerde benzer şekilde bozulur; sonradan eklenmiş/kopyalanmış
    bölgeler farklı sıkıştırma geçmişine sahip olduğundan öne çıkar.
    """
    rgb_image = image.convert("RGB")

    buffer = io.BytesIO()
    rgb_image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    diff = ImageChops.difference(rgb_image, resaved)

    # Farkı göz ile görülebilir hale getirmek için parlaklığı artırıyoruz.
    extrema = diff.getextrema()
    max_diff = max(channel_max for _, channel_max in extrema) or 1
    enhance_factor = min(scale, 255 / max_diff) if max_diff > 0 else scale
    ela_image = ImageEnhance.Brightness(diff).enhance(enhance_factor)

    # --- Skor hesaplama: YEREL anormallikleri arıyoruz ---
    # ELA'nın asıl değeri, görüntünün genelinden çok farklı davranan
    # KÜÇÜK BÖLGELERİ yakalamakta. Bu yüzden görüntüyü bloklara bölüp
    # her bloğun ortalama farkına bakıyoruz; ortalamadan aşırı sapan
    # bloklar "nokta atışı" bir manipülasyona işaret edebilir.
    gray_diff = np.asarray(diff.convert("L"), dtype=np.float32)

    block_size = max(8, min(gray_diff.shape) // 25)
    h, w = gray_diff.shape
    h_trim = (h // block_size) * block_size
    w_trim = (w // block_size) * block_size
    trimmed = gray_diff[:h_trim, :w_trim]

    blocks = trimmed.reshape(
        h_trim // block_size, block_size, w_trim // block_size, block_size
    )
    block_means = blocks.mean(axis=(1, 3))

    overall_mean = float(block_means.mean())
    # Standart sapmayı taban bir değerin altına düşürmüyoruz — aksi halde
    # zaten çok temiz/düz bir görüntüde (fark ~0'a yakın) en ufak bir
    # dalgalanma bile "aşırı sapma" gibi görünüp yanlış alarm üretiyor.
    overall_std = max(float(block_means.std()), 1.5)
    max_block_mean = float(block_means.max())

    # Bir bloğun "aşırı uç" (outlier) sayılması için HEM istatistiksel
    # olarak ortalamadan belirgin şekilde sapması HEM DE mutlak olarak
    # görülebilir bir fark taşıması gerekiyor (ör. 0.1 ile 0.3 arasındaki
    # anlamsız gürültü farkları z-score'da şişse bile filtrelenir).
    z_scores = (block_means - overall_mean) / overall_std
    is_outlier = (z_scores > 3.0) & (block_means > 6.0)
    outlier_ratio = float(is_outlier.mean())

    spike_signal = min(max(0.0, (max_block_mean - overall_mean - 6.0) / (overall_std * 5)), 1.0)
    ela_score = int(max(0, min(1, spike_signal * 0.6 + outlier_ratio * 20 * 0.4)) * 100)

    mean_diff_val = float(gray_diff.mean())

    stats = {
        "mean_difference": round(mean_diff_val, 2),
        "high_difference_ratio": round(outlier_ratio * 100, 2),
        "ela_score": ela_score,
    }

    return ela_image, stats


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# Yaygın telefon ekran çözünürlükleri (genellikle screenshot'lar bu boyutlarda
# çıkar çünkü tam olarak ekranın piksel boyutuna eşittir).
COMMON_SCREEN_RESOLUTIONS = {
    (1080, 1920), (1080, 2340), (1080, 2400), (1080, 2412), (1080, 2280),
    (828, 1792), (750, 1334), (1170, 2532), (1179, 2556), (1284, 2778),
    (1290, 2796), (1440, 2960), (1440, 3040), (1440, 3120), (1440, 3200),
    (720, 1280), (1200, 2000), (1200, 2670), (1080, 2160), (1242, 2688),
    (1125, 2436), (1284, 2778), (1600, 2560), (1350, 2400),
}


def detect_screenshot(image: Image.Image, filename: str, exif_data: dict) -> dict:
    """
    Ekran görüntüsü olma ihtimalini kaba ipuçlarıyla tahmin eder.
    Kesin bir tespit değil, sadece ELA/metadata güvenilirliğini
    yorumlarken kullanıcıyı uyarmak için bir sinyal.
    """
    reasons = []

    name_lower = (filename or "").lower()
    if any(k in name_lower for k in ["screenshot", "ekran", "ekrangörüntüsü", "screen_shot", "screen shot"]):
        reasons.append("Dosya adı ekran görüntüsüne işaret ediyor.")

    w, h = image.size
    normalized = (w, h) if w <= h else (h, w)
    is_common_res = normalized in {
        (min(a, b), max(a, b)) for a, b in COMMON_SCREEN_RESOLUTIONS
    }
    if is_common_res and not exif_data:
        reasons.append("Görsel boyutu yaygın bir telefon ekran çözünürlüğüyle eşleşiyor ve EXIF verisi yok.")

    if image.format == "PNG" and not exif_data:
        reasons.append("PNG formatında ve EXIF verisi yok — kameralar genelde JPEG üretir, PNG genellikle ekran görüntüsü/düzenleme çıktısıdır.")

    is_likely_screenshot = len(reasons) >= 1
    confidence = "yüksek" if len(reasons) >= 2 else "düşük"

    return {
        "is_likely_screenshot": is_likely_screenshot,
        "confidence": confidence if is_likely_screenshot else None,
        "reasons": reasons,
    }


def detect_copy_move(image: Image.Image, block_size: int = 12, stride: int = 1) -> dict:
    """
    Blok tabanlı copy-move (kopyala-yapıştır) tespiti.

    Mantık: Görüntüyü küçük, üst üste binen bloklara ayırıyoruz (numpy ile
    vektörize ederek, HER piksel konumundan — stride=1 — çünkü gerçek bir
    kopyala-yapıştırdaki kayma miktarı rastgele bir tam sayıdır, belirli
    bir adıma denk gelmeyebilir). Her blok için kaba bir "parmak izi"
    (4x4'e indirgenmiş, kuantize edilmiş bir özet) çıkarıyoruz. Aynı
    parmak izine sahip ama birbirinden yeterince UZAK iki blok bulursak,
    bunlar muhtemelen aynı bölgenin kopyası. Rastgele tesadüfleri elemek
    için, çok sayıda blok çiftinin AYNI kayma vektörüne (dx, dy) sahip
    olmasını arıyoruz — gerçek bir kopyala-yapıştırda, kopyalanan tüm
    bloklar aynı yöne/miktara kaymış olur, bu da güçlü bir istatistiksel
    sinyal verir. Hesaplama maliyetini düşük tutmak için görüntüyü önce
    küçültüyoruz.
    """
    original_size = image.size
    working = image.convert("L")

    max_dim = 260
    scale = min(1.0, max_dim / max(working.size))
    if scale < 1.0:
        working = working.resize(
            (max(1, int(working.width * scale)), max(1, int(working.height * scale))),
            Image.BILINEAR,
        )

    arr = np.asarray(working, dtype=np.float32)
    h, w = arr.shape

    if h < block_size * 3 or w < block_size * 3:
        return {"detected": False, "reason": "Görüntü bu analiz için çok küçük."}

    windows = np.lib.stride_tricks.sliding_window_view(arr, (block_size, block_size))
    ny, nx = windows.shape[:2]

    stds = windows.std(axis=(2, 3))
    sample_step = max(1, block_size // 4)
    small = windows[:, :, ::sample_step, ::sample_step][:, :, :4, :4]
    feats = small.reshape(ny, nx, -1)
    quant = (feats // 8).astype(np.int16)

    valid_mask = stds >= 7.0

    buckets: dict = {}
    valid_ys, valid_xs = np.nonzero(valid_mask)
    for iy, ix in zip(valid_ys.tolist(), valid_xs.tolist()):
        key = tuple(quant[iy, ix].tolist())
        buckets.setdefault(key, []).append((iy, ix))

    MIN_DISTANCE_PX = block_size * 2.5
    offset_votes: dict = {}

    for key, cells in buckets.items():
        if len(cells) < 2 or len(cells) > 25:
            continue
        for a in range(len(cells)):
            for b in range(a + 1, len(cells)):
                y1, x1 = cells[a]
                y2, x2 = cells[b]

                dist = ((y1 - y2) ** 2 + (x1 - x2) ** 2) ** 0.5
                if dist < MIN_DISTANCE_PX:
                    continue

                block1 = windows[y1, x1]
                block2 = windows[y2, x2]
                mad = float(np.abs(block1.astype(np.float32) - block2.astype(np.float32)).mean())
                if mad > 6.0:
                    continue

                dx, dy = x2 - x1, y2 - y1
                offset_key = (round(dx / 3) * 3, round(dy / 3) * 3)
                offset_votes.setdefault(offset_key, []).append(((y1, x1), (y2, x2)))

    if not offset_votes:
        return {"detected": False, "reason": None}

    best_offset, matches = max(offset_votes.items(), key=lambda kv: len(kv[1]))

    MIN_MATCHES = 20
    if len(matches) < MIN_MATCHES:
        return {"detected": False, "reason": None}

    source_pts = [m[0] for m in matches]
    target_pts = [m[1] for m in matches]

    def bbox(points):
        ys = [p[0] for p in points]
        xs = [p[1] for p in points]
        return (min(xs), min(ys), max(xs) + block_size, max(ys) + block_size)

    src_box = bbox(source_pts)
    tgt_box = bbox(target_pts)

    inv_scale = 1.0 / scale if scale < 1.0 else 1.0
    src_box_orig = tuple(int(v * inv_scale) for v in src_box)
    tgt_box_orig = tuple(int(v * inv_scale) for v in tgt_box)

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    line_w = max(3, original_size[0] // 200)
    draw.rectangle(src_box_orig, outline=(239, 68, 68), width=line_w)
    draw.rectangle(tgt_box_orig, outline=(59, 130, 246), width=line_w)

    return {
        "detected": True,
        "match_count": len(matches),
        "source_box": src_box_orig,
        "target_box": tgt_box_orig,
        "annotated_image_base64": image_to_base64(annotated),
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

    ela_image, ela_stats = compute_ela(image)
    ela_image_b64 = image_to_base64(ela_image)

    screenshot_check = detect_screenshot(image, file.filename, exif_data)
    copy_move_result = detect_copy_move(image)

    # Genel şüphe skoru: metadata + ELA + copy-move sinyallerinin birleşimi.
    copy_move_score = 70 if copy_move_result.get("detected") else 0
    overall_score = round(
        metadata_analysis["suspicion_score"] * 0.25
        + ela_stats["ela_score"] * 0.35
        + copy_move_score * 0.4
    )

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
        "ela_analysis": {
            **ela_stats,
            "image_base64": ela_image_b64,
        },
        "copy_move_analysis": copy_move_result,
        "screenshot_check": screenshot_check,
        "overall_suspicion_score": overall_score,
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

  .screenshot-warning {
    display: none;
    background: #4a2a0022;
    border: 1px solid #ff6b3555;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 16px;
    font-size: 0.85rem;
  }
  .screenshot-warning strong { color: var(--accent); display: block; margin-bottom: 4px; }
  .screenshot-warning ul { margin: 6px 0 0; padding-left: 18px; color: var(--muted); }

  .ela-desc { color: var(--muted); font-size: 0.82rem; margin: 0 0 10px; }
  .ela-img { width: 100%; border-radius: 10px; border: 1px solid var(--border); display: block; margin-bottom: 10px; }

  .cm-legend { display: flex; gap: 16px; font-size: 0.82rem; color: var(--muted); }
  .cm-legend span { display: flex; align-items: center; gap: 6px; }
  .cm-dot { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }

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
    <div class="screenshot-warning" id="screenshotWarning">
      <strong>⚠️ Bu görsel ekran görüntüsü olabilir</strong>
      Ekran görüntülerinde orijinal metadata kaybolur ve ELA analizi daha az güvenilir hale gelir. Mümkünse fotoğrafı orijinal dosya olarak yükleyin.
      <ul id="screenshotReasons"></ul>
    </div>

    <div class="score-card">
      <div class="score-num" id="scoreNum">0</div>
      <div class="score-label">Genel Şüphe Skoru / 100</div>
    </div>

    <div class="section">
      <h3>Bulgular</h3>
      <div id="flagsList"></div>
    </div>

    <div class="section">
      <h3>ELA — Yeniden Sıkıştırma Analizi</h3>
      <p class="ela-desc">Parlak/farklı görünen bölgeler, fotoğrafın geri kalanından farklı bir düzenleme geçmişine sahip olabilir.</p>
      <img id="elaImage" class="ela-img" alt="ELA görseli">
      <div id="elaStats"></div>
    </div>

    <div class="section" id="copyMoveSection" style="display:none;">
      <h3>Copy-Move Tespiti</h3>
      <p class="ela-desc" id="copyMoveDesc"></p>
      <img id="copyMoveImage" class="ela-img" alt="Copy-move görseli">
      <div id="copyMoveLegend" class="cm-legend"></div>
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
  const ela = data.ela_analysis;
  const overallScore = data.overall_suspicion_score;
  const screenshotCheck = data.screenshot_check;

  const warningBox = document.getElementById("screenshotWarning");
  if (screenshotCheck && screenshotCheck.is_likely_screenshot) {
    warningBox.style.display = "block";
    const reasonsList = document.getElementById("screenshotReasons");
    reasonsList.innerHTML = "";
    screenshotCheck.reasons.forEach(r => {
      const li = document.createElement("li");
      li.textContent = r;
      reasonsList.appendChild(li);
    });
  } else {
    warningBox.style.display = "none";
  }

  document.getElementById("scoreNum").textContent = overallScore;
  document.getElementById("scoreNum").style.color =
    overallScore >= 50 ? "#e5484d" :
    overallScore >= 20 ? "#ff6b35" : "#3ecf8e";

  const flagsList = document.getElementById("flagsList");
  flagsList.innerHTML = "";
  const allFlags = [...analysis.flags];
  if (ela.ela_score >= 40) {
    allFlags.push(`ELA analizinde yüksek fark oranı tespit edildi (skor: ${ela.ela_score}/100) — görselin bazı bölgeleri farklı bir sıkıştırma geçmişine sahip olabilir.`);
  }
  if (data.copy_move_analysis && data.copy_move_analysis.detected) {
    allFlags.push(`Kopyala-yapıştır şüphesi: görselde birbirinin neredeyse birebir aynısı olan iki bölge bulundu (${data.copy_move_analysis.match_count} eşleşen blok).`);
  }
  if (allFlags.length === 0) {
    const d = document.createElement("div");
    d.className = "flag clean";
    d.textContent = "Belirgin bir şüphe işareti bulunamadı.";
    flagsList.appendChild(d);
  } else {
    allFlags.forEach(f => {
      const d = document.createElement("div");
      d.className = "flag";
      d.textContent = f;
      flagsList.appendChild(d);
    });
  }

  document.getElementById("elaImage").src = "data:image/png;base64," + ela.image_base64;
  const elaStats = document.getElementById("elaStats");
  elaStats.innerHTML = "";
  [
    ["ELA skoru", ela.ela_score + " / 100"],
    ["Ortalama fark", ela.mean_difference],
    ["Yüksek fark oranı", ela.high_difference_ratio + "%"],
  ].forEach(([label, val]) => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<span>${label}</span><span>${val}</span>`;
    elaStats.appendChild(row);
  });

  const cm = data.copy_move_analysis;
  const cmSection = document.getElementById("copyMoveSection");
  if (cm && cm.detected) {
    cmSection.style.display = "block";
    document.getElementById("copyMoveDesc").textContent =
      `Kopyalanmış olabilecek bir bölge tespit edildi (${cm.match_count} eşleşen blok). Kırmızı: kaynak bölge, Mavi: hedef (yapıştırılan) bölge.`;
    document.getElementById("copyMoveImage").src = "data:image/png;base64," + cm.annotated_image_base64;
    document.getElementById("copyMoveLegend").innerHTML =
      '<span><span class="cm-dot" style="background:#ef4444"></span>Kaynak</span>' +
      '<span><span class="cm-dot" style="background:#3b82f6"></span>Hedef</span>';
  } else {
    cmSection.style.display = "none";
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

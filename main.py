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
from PIL import Image, ImageChops, ImageEnhance, ImageDraw, ImageFilter
from PIL.ExifTags import TAGS, GPSTAGS
from scipy import ndimage

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

    # Büyük orijinal fotoğraflarda (ör. 4000x3000) JPEG yeniden kaydetme +
    # piksel farkı hesaplama gereksiz yere yavaşlıyor; analiz için makul
    # bir üst sınır yeterli ve sonucu pratikte değiştirmiyor.
    max_dim = 1200
    if max(rgb_image.size) > max_dim:
        ratio = max_dim / max(rgb_image.size)
        new_size = (max(1, int(rgb_image.width * ratio)), max(1, int(rgb_image.height * ratio)))
        rgb_image = rgb_image.resize(new_size, Image.BILINEAR)

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


def image_to_base64(image: Image.Image, max_dim: int = 900) -> str:
    """
    Görseli base64 PNG'ye çevirir. Görüntüleme amaçlı olduğu için, büyük
    orijinal fotoğrafları (ör. 4000x3000) olduğu gibi kodlamak hem PNG
    sıkıştırmasını çok yavaşlatır hem de mobilde çok büyük bir veri
    indirmesine yol açar — bu yüzden önce makul bir boyuta küçültüyoruz.
    Analiz sonuçları zaten küçültülmüş görseller üzerinden hesaplandığı
    için, bu sadece görüntüleme kalitesini etkiler, doğruluğu değil.
    """
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
        image = image.resize(new_size, Image.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=False, compress_level=4)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _prepare_display_copy(image: Image.Image, max_dim: int = 900) -> tuple:
    """
    İşaretleme (kutu çizme) için orijinal görselin KÜÇÜLTÜLMÜŞ bir
    kopyasını hazırlar. Büyük orijinal fotoğraflarda (ör. 4000x3000)
    doğrudan tam boyutta çizim yapıp sonra küçültmek çok yavaş oluyordu;
    bunun yerine önce küçültüp öyle çiziyoruz. Dönen `scale` değeri,
    ORİJİNAL görseldeki koordinatları bu küçük kopyadaki koordinatlara
    çevirmek için çarpılacak katsayıdır.
    """
    display = image.convert("RGB")
    scale = 1.0
    if max(display.size) > max_dim:
        scale = max_dim / max(display.size)
        new_size = (max(1, int(display.width * scale)), max(1, int(display.height * scale)))
        display = display.resize(new_size, Image.BILINEAR)
    return display, scale


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

    # --- Yoğunluk kontrolü (asfalt/çim/su gibi tekrarlayan dokularda
    # yanlış pozitifi engellemek için) ---
    # Gerçek bir kopyala-yapıştırda, kaynak dikdörtgenin İÇİ neredeyse
    # tamamen eşleşen bloklarla dolu olur (yoğun/sıkı bir küme). Tekrarlayan
    # bir doku (asfalt, çim, su vb.) ise, görüntünün geniş bir alanına
    # YAYILMIŞ, seyrek/dağınık eşleşmeler üretir — eşleşme sayısı yüksek
    # olsa bile. Bu ikisini ayırt etmek için, bulunan bölgenin alanına göre
    # eşleşme yoğunluğuna bakıyoruz.
    src_w = src_box[2] - src_box[0]
    src_h = src_box[3] - src_box[1]
    expected_positions = max(1, (src_w * src_h) / (block_size * block_size))
    density = len(matches) / expected_positions

    MIN_DENSITY = 0.12
    if density < MIN_DENSITY:
        return {"detected": False, "reason": None}

    inv_scale = 1.0 / scale if scale < 1.0 else 1.0
    src_box_orig = tuple(int(v * inv_scale) for v in src_box)
    tgt_box_orig = tuple(int(v * inv_scale) for v in tgt_box)

    # Çizimi orijinal boyutta değil, küçük bir "gösterim kopyası" üzerinde
    # yapıyoruz (büyük fotoğraflarda ciddi hız kazandırıyor).
    display, display_scale = _prepare_display_copy(image)
    src_box_display = tuple(int(v * display_scale) for v in src_box_orig)
    tgt_box_display = tuple(int(v * display_scale) for v in tgt_box_orig)

    draw = ImageDraw.Draw(display)
    line_w = max(2, display.size[0] // 200)
    draw.rectangle(src_box_display, outline=(239, 68, 68), width=line_w)
    draw.rectangle(tgt_box_display, outline=(59, 130, 246), width=line_w)

    return {
        "detected": True,
        "match_count": len(matches),
        "source_box": src_box_orig,
        "target_box": tgt_box_orig,
        "annotated_image_base64": image_to_base64(display),
    }


def detect_objects_and_check_geometry(image: Image.Image) -> dict:
    """
    Hafif (YOLO gerektirmeyen) nesne sayımı + tutarlılık kontrolü.

    Yaklaşım: Kenar tespiti yapıp, birbirine bitişik kenar bölgelerini
    "nesne" adayı olarak grupluyoruz (connected components). Sonra:
    1. Kaç nesne bulundu, boyutları
    2. Nesne boyutları birbirinden aşırı farklı mı (biri diğerlerinden
       çok daha büyük/küçükse bu "eklenmiş/farklı kaynaklı" bir nesne
       olabilir — ya da sadece perspektiften kaynaklanan doğal bir fark
       olabilir; bu yüzden bunu KESİN bir sinyal değil, dikkat çekici
       bir gözlem olarak sunuyoruz)
    3. Her nesnenin hemen altındaki bölgenin ortalama parlaklığına bakıp
       kaba bir "gölge yönü" tahmini yapıyoruz, nesneler arası
       tutarsızlık var mı diye bakıyoruz

    NOT: Bu YOLO gibi "bu bir kutu" diyebilen akıllı bir tespit değil,
    sadece kenar yoğunluğuna dayalı kaba bir bölgeleme. Amaç, bariz
    tutarsızlıkları yakalamak; kesin bir sayım garantisi vermiyor.
    """
    original_size = image.size
    gray = image.convert("L")

    max_dim = 500
    scale = min(1.0, max_dim / max(gray.size))
    if scale < 1.0:
        gray = gray.resize(
            (max(1, int(gray.width * scale)), max(1, int(gray.height * scale))),
            Image.BILINEAR,
        )

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.asarray(edges, dtype=np.float32)

    threshold = max(20.0, float(np.percentile(edge_arr, 85)))
    binary = edge_arr > threshold
    # FIND_EDGES filtresi görüntünün en dış çerçevesinde yapay/sahte bir
    # "kenar" oluşturur (gerçek bir kenar değil, filtrenin sınır etkisi).
    # Bu, delik doldurma mantığını bozduğu için dış çerçeveyi temizliyoruz.
    binary[0, :] = False
    binary[-1, :] = False
    binary[:, 0] = False
    binary[:, -1] = False

    # Kenar parçalarını birbirine yakınsa birleştir (nesnenin dış hattı
    # kesintili çıkabiliyor, bunu kapatıyoruz).
    structure = np.ones((3, 3))
    closed = ndimage.binary_closing(binary, structure=structure, iterations=3)
    closed = ndimage.binary_fill_holes(closed)

    labeled, num_features = ndimage.label(closed, structure=structure)

    if num_features == 0:
        return {"object_count": 0, "objects": [], "size_warning": None, "shadow_warning": None}

    h, w = edge_arr.shape
    min_area = (h * w) * 0.003   # çok küçük gürültü parçalarını ele
    max_area = (h * w) * 0.5     # görüntünün yarısından büyükse muhtemelen arka plandır, nesne değil

    objects = []
    slices = ndimage.find_objects(labeled)
    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        area = int((labeled[sl] == i).sum())
        if area < min_area or area > max_area:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        box_w, box_h = x1 - x0, y1 - y0
        if box_w < 10 or box_h < 10:
            continue
        objects.append({"box": (x0, y0, x1, y1), "area": area})

    if not objects:
        return {"object_count": 0, "objects": [], "size_warning": None, "shadow_warning": None}

    # Boyut tutarlılığı: alanların medyanından aşırı sapan var mı?
    areas = np.array([o["area"] for o in objects], dtype=np.float32)
    median_area = float(np.median(areas))
    size_warning = None
    outlier_indices = []
    for idx, a in enumerate(areas):
        if a > median_area * 3 or a < median_area / 3:
            outlier_indices.append(idx)
    if outlier_indices and len(objects) >= 3:
        size_warning = (
            f"{len(outlier_indices)} nesnenin boyutu diğerlerinden belirgin şekilde "
            f"farklı — bu perspektiften kaynaklanabilir ya da farklı bir kaynaktan "
            f"eklenmiş olabilir, tek başına kesin bir kanıt değildir."
        )

    # NOT: Gölge yönü tutarlılığı kontrolünü de denedik, ama sentetik
    # testlerde güvenilir sonuç vermedi (gerçek gölgeler basit test
    # şekillerinden çok daha karmaşık). Yanlış/yanıltıcı bir sinyal
    # vermemek için bu özelliği şimdilik çıkardık — ileride daha sağlam
    # bir yöntemle (örn. ışık kaynağı modelleme) yeniden ele alınabilir.
    shadow_warning = None

    # Görselleştirme (küçük bir gösterim kopyası üzerinde, büyük fotoğraflarda hız için)
    inv_scale = 1.0 / scale if scale < 1.0 else 1.0
    display, display_scale = _prepare_display_copy(image)
    draw = ImageDraw.Draw(display)
    line_w = max(2, display.size[0] // 300)
    for idx, o in enumerate(objects, start=1):
        x0, y0, x1, y1 = o["box"]
        box_orig = tuple(v * inv_scale for v in (x0, y0, x1, y1))
        box_display = tuple(int(v * display_scale) for v in box_orig)
        color = (250, 204, 21) if idx - 1 not in outlier_indices else (239, 68, 68)
        draw.rectangle(box_display, outline=color, width=line_w)

    return {
        "object_count": len(objects),
        "size_warning": size_warning,
        "shadow_warning": shadow_warning,
        "annotated_image_base64": image_to_base64(display),
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
    geometry_result = detect_objects_and_check_geometry(image)

    # Genel şüphe skoru: metadata + ELA + copy-move + geometri sinyalleri.
    copy_move_score = 70 if copy_move_result.get("detected") else 0
    geometry_score = 0
    if geometry_result.get("size_warning"):
        geometry_score += 25
    if geometry_result.get("shadow_warning"):
        geometry_score += 25

    weights = {
        "metadata": 0.2,
        "ela": 0.3,
        "copy_move": 0.35,
        "geometry": 0.15,
    }
    raw_scores = {
        "metadata": metadata_analysis["suspicion_score"],
        "ela": ela_stats["ela_score"],
        "copy_move": copy_move_score,
        "geometry": geometry_score,
    }
    overall_score = round(sum(raw_scores[k] * weights[k] for k in weights))

    signal_breakdown = [
        {
            "label": "Metadata / EXIF",
            "raw_score": raw_scores["metadata"],
            "weight_pct": int(weights["metadata"] * 100),
            "contribution": round(raw_scores["metadata"] * weights["metadata"], 1),
        },
        {
            "label": "ELA (sıkıştırma analizi)",
            "raw_score": raw_scores["ela"],
            "weight_pct": int(weights["ela"] * 100),
            "contribution": round(raw_scores["ela"] * weights["ela"], 1),
        },
        {
            "label": "Copy-Move tespiti",
            "raw_score": raw_scores["copy_move"],
            "weight_pct": int(weights["copy_move"] * 100),
            "contribution": round(raw_scores["copy_move"] * weights["copy_move"], 1),
        },
        {
            "label": "Nesne/geometri tutarlılığı",
            "raw_score": raw_scores["geometry"],
            "weight_pct": int(weights["geometry"] * 100),
            "contribution": round(raw_scores["geometry"] * weights["geometry"], 1),
        },
    ]

    if overall_score >= 60:
        verdict = "Yüksek şüphe"
        verdict_detail = "Birden fazla güçlü sinyal bir arada bulundu. Bu görselin dikkatle incelenmesini öneririz."
    elif overall_score >= 30:
        verdict = "Orta düzey şüphe"
        verdict_detail = "Bazı dikkat çekici bulgular var, ama tek başına kesin bir manipülasyon kanıtı değil. Bulgular kısmını inceleyin."
    else:
        verdict = "Düşük şüphe"
        verdict_detail = "Belirgin bir manipülasyon sinyali bulunamadı. Bu, görselin kesinlikle orijinal olduğu anlamına gelmez, sadece bu analizlerin bir sorun tespit etmediği anlamına gelir."

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
        "geometry_analysis": geometry_result,
        "screenshot_check": screenshot_check,
        "signal_breakdown": signal_breakdown,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
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
<title>V-ERA — Görsel Sahtecilik Tespit Sistemi</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0c0f;
    --panel: #12151a;
    --panel-raised: #171b21;
    --border: #232830;
    --text: #eef0f2;
    --muted: #8a92a0;
    --amber: #e8a33d;
    --amber-dim: #e8a33d1a;
    --cyan: #4dc4d9;
    --danger: #ef4444;
    --ok: #22c55e;
    --source: #ef4444;
    --target: #3b82f6;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: 'Inter', -apple-system, "Segoe UI", sans-serif;
    background: var(--bg);
    background-image:
      radial-gradient(ellipse 600px 300px at 50% -10%, #e8a33d0d, transparent);
    color: var(--text);
    padding: 28px 16px 60px;
    line-height: 1.5;
  }
  .wrap { max-width: 620px; margin: 0 auto; }

  .brand {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 4px;
  }
  .brand-mark {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    letter-spacing: -0.02em;
    color: var(--text);
  }
  .brand-mark span { color: var(--amber); }
  .tagline { color: var(--muted); font-size: 0.88rem; margin-bottom: 28px; }

  /* --- Upload zone: kanıt yükleme alanı, köşe işaretleriyle bir tarayıcı/vizör hissi --- */
  .dropzone {
    position: relative;
    border: 1.5px dashed var(--border);
    border-radius: 16px;
    padding: 44px 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.18s ease, background 0.18s ease;
    background: var(--panel);
  }
  .dropzone.drag { border-color: var(--amber); background: var(--amber-dim); }
  .dropzone::before, .dropzone::after,
  .corner-tl, .corner-br { display: none; }
  .scan-corner {
    position: absolute;
    width: 18px;
    height: 18px;
    border: 2px solid var(--border);
    opacity: 0.9;
    transition: border-color 0.18s ease;
  }
  .dropzone.drag .scan-corner { border-color: var(--amber); }
  .scan-corner.tl { top: 10px; left: 10px; border-right: none; border-bottom: none; border-top-left-radius: 6px; }
  .scan-corner.tr { top: 10px; right: 10px; border-left: none; border-bottom: none; border-top-right-radius: 6px; }
  .scan-corner.bl { bottom: 10px; left: 10px; border-right: none; border-top: none; border-bottom-left-radius: 6px; }
  .scan-corner.br { bottom: 10px; right: 10px; border-left: none; border-top: none; border-bottom-right-radius: 6px; }
  .dropzone p { margin: 8px 0 0; color: var(--muted); font-size: 0.85rem; }
  .dropzone strong { color: var(--text); font-size: 1.02rem; font-weight: 600; }
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
    padding: 15px;
    border: none;
    border-radius: 10px;
    background: var(--amber);
    color: #14110a;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    letter-spacing: -0.01em;
    cursor: pointer;
    transition: transform 0.1s ease, opacity 0.15s ease;
  }
  button.analyze:active { transform: scale(0.98); }
  button.analyze:disabled { opacity: 0.5; }

  .report {
    margin-top: 28px;
    display: none;
  }

  /* --- Skor göstergesi: dairesel gösterge (gauge) --- */
  .score-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 28px 20px;
    text-align: center;
    margin-bottom: 16px;
  }
  .gauge-wrap { position: relative; width: 148px; height: 148px; margin: 0 auto; }
  .gauge-wrap svg { width: 100%; height: 100%; transform: rotate(-90deg); }
  .gauge-track { fill: none; stroke: var(--border); stroke-width: 10; }
  .gauge-fill {
    fill: none;
    stroke-width: 10;
    stroke-linecap: round;
    transition: stroke-dashoffset 0.8s cubic-bezier(0.16, 1, 0.3, 1), stroke 0.3s ease;
  }
  .gauge-center {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }
  .score-num { font-family: 'Space Grotesk', sans-serif; font-size: 2.6rem; font-weight: 700; line-height: 1; }
  .score-label { color: var(--muted); font-size: 0.72rem; margin-top: 2px; }
  .verdict { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.1rem; margin-top: 18px; }
  .verdict-detail { color: var(--muted); font-size: 0.85rem; margin-top: 6px; line-height: 1.5; max-width: 440px; margin-left: auto; margin-right: auto; }

  .signal-row { margin-bottom: 14px; }
  .signal-row:last-child { margin-bottom: 0; }
  .signal-row-top { display: flex; justify-content: space-between; align-items: baseline; font-size: 0.85rem; margin-bottom: 5px; }
  .signal-row-top .signal-label { color: var(--text); font-weight: 500; }
  .signal-row-top .signal-weight { color: var(--muted); font-size: 0.78rem; }
  .signal-row-top .signal-value { color: var(--muted); font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; }
  .signal-bar-bg { height: 5px; border-radius: 3px; background: var(--border); overflow: hidden; }
  .signal-bar-fill { height: 100%; border-radius: 3px; background: var(--amber); }

  .screenshot-warning {
    display: none;
    background: #e8a33d14;
    border: 1px solid #e8a33d40;
    border-left: 3px solid var(--amber);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 16px;
    font-size: 0.85rem;
  }
  .screenshot-warning strong { color: var(--amber); display: block; margin-bottom: 4px; font-weight: 600; }
  .screenshot-warning ul { margin: 6px 0 0; padding-left: 18px; color: var(--muted); }

  .section-desc { color: var(--muted); font-size: 0.85rem; margin: 0 0 12px; }
  .section-img { width: 100%; border-radius: 10px; border: 1px solid var(--border); display: block; margin-bottom: 10px; }

  .cm-legend { display: flex; gap: 18px; font-size: 0.82rem; color: var(--muted); }
  .cm-legend span { display: flex; align-items: center; gap: 6px; }
  .cm-dot { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }

  /* --- Bölümler: kategoriye göre renkli sol şerit, küçük harf başlıklar --- */
  .section {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--cat-color, var(--border));
    border-radius: 4px 14px 14px 4px;
    padding: 18px 20px;
    margin-bottom: 12px;
  }
  .section.cat-meta { --cat-color: #6b7280; }
  .section.cat-score { --cat-color: var(--amber); }
  .section.cat-ela { --cat-color: var(--cyan); }
  .section.cat-copymove { --cat-color: #f472b6; }
  .section.cat-geometry { --cat-color: #a78bfa; }

  .section h3 {
    margin: 0 0 12px;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text);
    font-family: 'Space Grotesk', sans-serif;
  }
  .row {
    display: flex;
    justify-content: space-between;
    padding: 7px 0;
    font-size: 0.88rem;
    border-bottom: 1px solid var(--border);
  }
  .row:last-child { border-bottom: none; }
  .row span:first-child { color: var(--muted); }

  .flag {
    display: flex;
    gap: 9px;
    padding: 9px 0;
    font-size: 0.87rem;
    align-items: flex-start;
    border-bottom: 1px solid var(--border);
  }
  .flag:last-child { border-bottom: none; }
  .flag::before { content: "●"; color: var(--amber); flex-shrink: 0; font-size: 0.6rem; margin-top: 6px; }
  .flag.clean::before { content: "●"; color: var(--ok); }

  #status { text-align: center; color: var(--muted); font-size: 0.85rem; margin-top: 12px; display: none; }
  #status::before { content: "◌ "; }
</style>
</head>
<body>
<div class="wrap">
  <div class="brand">
    <div class="brand-mark">V-<span>ERA</span></div>
  </div>
  <div class="tagline">Görüntünün ötesine bakar: izleri inceler, manipülasyonları tespit eder.</div>

  <div class="dropzone" id="dropzone">
    <span class="scan-corner tl"></span>
    <span class="scan-corner tr"></span>
    <span class="scan-corner bl"></span>
    <span class="scan-corner br"></span>
    <strong>Fotoğraf seç ya da sürükle</strong>
    <p>JPG, PNG desteklenir</p>
    <input type="file" id="fileInput" accept="image/*">
  </div>

  <div id="preview"><img id="previewImg" alt="önizleme"></div>
  <button class="analyze" id="analyzeBtn" style="display:none;">Analiz et</button>
  <div id="status">Analiz ediliyor</div>

  <div class="report" id="report">
    <div class="screenshot-warning" id="screenshotWarning">
      <strong>Bu görsel ekran görüntüsü olabilir</strong>
      Ekran görüntülerinde orijinal metadata kaybolur ve ELA analizi daha az güvenilir hale gelir. Mümkünse fotoğrafı orijinal dosya olarak yükleyin.
      <ul id="screenshotReasons"></ul>
    </div>

    <div class="score-card">
      <div class="gauge-wrap">
        <svg viewBox="0 0 120 120">
          <circle class="gauge-track" cx="60" cy="60" r="52"></circle>
          <circle class="gauge-fill" id="gaugeFill" cx="60" cy="60" r="52"></circle>
        </svg>
        <div class="gauge-center">
          <div class="score-num" id="scoreNum">0</div>
          <div class="score-label">/ 100</div>
        </div>
      </div>
      <div class="verdict" id="verdictText"></div>
      <div class="verdict-detail" id="verdictDetail"></div>
    </div>

    <div class="section cat-score">
      <h3>Skor dökümü</h3>
      <p class="section-desc">Genel skor dört farklı sinyalin ağırlıklı ortalamasıdır. Her birinin ne kadar etkili olduğu aşağıda.</p>
      <div id="signalBreakdown"></div>
    </div>

    <div class="section cat-meta">
      <h3>Bulgular</h3>
      <div id="flagsList"></div>
    </div>

    <div class="section cat-ela">
      <h3>ELA — yeniden sıkıştırma analizi</h3>
      <p class="section-desc">Parlak/farklı görünen bölgeler, fotoğrafın geri kalanından farklı bir düzenleme geçmişine sahip olabilir.</p>
      <img id="elaImage" class="section-img" alt="ELA görseli">
      <div id="elaStats"></div>
    </div>

    <div class="section cat-copymove" id="copyMoveSection" style="display:none;">
      <h3>Copy-move tespiti</h3>
      <p class="section-desc" id="copyMoveDesc"></p>
      <img id="copyMoveImage" class="section-img" alt="Copy-move görseli">
      <div id="copyMoveLegend" class="cm-legend"></div>
    </div>

    <div class="section cat-geometry" id="geometrySection" style="display:none;">
      <h3>Nesne sayımı ve tutarlılık</h3>
      <p class="section-desc" id="geometryCount"></p>
      <img id="geometryImage" class="section-img" alt="Nesne tespiti görseli">
      <div id="geometryWarnings"></div>
    </div>

    <div class="section cat-meta">
      <h3>Dosya bilgisi</h3>
      <div id="fileInfo"></div>
    </div>
  </div>
</div>

<script>
const API_URL = window.location.origin + "/analyze";
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 52;

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const previewImg = document.getElementById("previewImg");
const analyzeBtn = document.getElementById("analyzeBtn");
const statusEl = document.getElementById("status");
const report = document.getElementById("report");

const gaugeFill = document.getElementById("gaugeFill");
gaugeFill.style.strokeDasharray = `${GAUGE_CIRCUMFERENCE}`;
gaugeFill.style.strokeDashoffset = `${GAUGE_CIRCUMFERENCE}`;

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

  const scoreColor = overallScore >= 60 ? "#ef4444" : overallScore >= 30 ? "#e8a33d" : "#22c55e";
  document.getElementById("scoreNum").textContent = overallScore;
  document.getElementById("scoreNum").style.color = scoreColor;

  const offset = GAUGE_CIRCUMFERENCE * (1 - overallScore / 100);
  gaugeFill.style.stroke = scoreColor;
  // Bir sonraki karede uygula ki geçiş animasyonu tetiklensin.
  requestAnimationFrame(() => {
    gaugeFill.style.strokeDashoffset = `${offset}`;
  });

  const verdictEl = document.getElementById("verdictText");
  verdictEl.textContent = data.verdict || "";
  verdictEl.style.color = scoreColor;
  document.getElementById("verdictDetail").textContent = data.verdict_detail || "";

  const breakdownEl = document.getElementById("signalBreakdown");
  breakdownEl.innerHTML = "";
  (data.signal_breakdown || []).forEach(s => {
    const row = document.createElement("div");
    row.className = "signal-row";
    row.innerHTML = `
      <div class="signal-row-top">
        <span class="signal-label">${s.label} <span class="signal-weight">(ağırlık %${s.weight_pct})</span></span>
        <span class="signal-value">${s.raw_score}/100</span>
      </div>
      <div class="signal-bar-bg"><div class="signal-bar-fill" style="width:${s.raw_score}%"></div></div>
    `;
    breakdownEl.appendChild(row);
  });

  const flagsList = document.getElementById("flagsList");
  flagsList.innerHTML = "";
  const allFlags = [...analysis.flags];
  if (ela.ela_score >= 40) {
    allFlags.push(`ELA analizinde yüksek fark oranı tespit edildi (skor: ${ela.ela_score}/100) — görselin bazı bölgeleri farklı bir sıkıştırma geçmişine sahip olabilir.`);
  }
  if (data.copy_move_analysis && data.copy_move_analysis.detected) {
    allFlags.push(`Kopyala-yapıştır şüphesi: görselde birbirinin neredeyse birebir aynısı olan iki bölge bulundu (${data.copy_move_analysis.match_count} eşleşen blok).`);
  }
  if (data.geometry_analysis) {
    if (data.geometry_analysis.size_warning) allFlags.push(data.geometry_analysis.size_warning);
    if (data.geometry_analysis.shadow_warning) allFlags.push(data.geometry_analysis.shadow_warning);
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
      `Kopyalanmış olabilecek bir bölge tespit edildi (${cm.match_count} eşleşen blok). Kırmızı: kaynak bölge, mavi: hedef (yapıştırılan) bölge.`;
    document.getElementById("copyMoveImage").src = "data:image/png;base64," + cm.annotated_image_base64;
    document.getElementById("copyMoveLegend").innerHTML =
      '<span><span class="cm-dot" style="background:#ef4444"></span>Kaynak</span>' +
      '<span><span class="cm-dot" style="background:#3b82f6"></span>Hedef</span>';
  } else {
    cmSection.style.display = "none";
  }

  const geo = data.geometry_analysis;
  const geoSection = document.getElementById("geometrySection");
  if (geo && geo.object_count > 0) {
    geoSection.style.display = "block";
    document.getElementById("geometryCount").textContent =
      `${geo.object_count} nesne tespit edildi. Sarı: normal, kırmızı: boyutu diğerlerinden belirgin farklı.`;
    document.getElementById("geometryImage").src = "data:image/png;base64," + geo.annotated_image_base64;
    const warnBox = document.getElementById("geometryWarnings");
    warnBox.innerHTML = "";
    [geo.size_warning, geo.shadow_warning].filter(Boolean).forEach(w => {
      const d = document.createElement("div");
      d.className = "flag";
      d.textContent = w;
      warnBox.appendChild(d);
    });
  } else {
    geoSection.style.display = "none";
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

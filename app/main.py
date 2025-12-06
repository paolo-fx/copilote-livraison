# app.py — Copilote Livraisons (OCR libellés + ROI ANJU, DnD, Carte, Exports)
# Version: 2025-09-10 — + propositions Patron SOUS LA CARTE (listes + appliquer)

import os, json, math, re
from io import StringIO, BytesIO
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from components.image_input import bon_image_picker


# >>>>> Tesseract (Windows) — pointez ici vers votre exe
import pytesseract
tess_env = os.getenv("TESSERACT_CMD")
if tess_env and Path(tess_env).exists():
    pytesseract.pytesseract.tesseract_cmd = tess_env
elif os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# sinon: on s'appuie sur le PATH

# <<<<<

# --------- Drag & Drop ----------
HAVE_DND = False
try:
    from streamlit_sortables import sort_items as _sort_items
    HAVE_DND = True
except Exception:
    HAVE_DND = False

def sort_items_safe(*args, **kwargs):
    try:
        return _sort_items(*args, **kwargs)
    except Exception:
        st.session_state["DND_BROKEN"] = True
        return kwargs.get("items", [])

# --------- Optionnels (PDF / géocodage / carte) ----------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    HAVE_REPORTLAB = True
except Exception:
    HAVE_REPORTLAB = False

try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    HAVE_GEOPY = True
except Exception:
    HAVE_GEOPY = False

try:
    import pydeck as pdk
    HAVE_PYDECK = True
except Exception:
    HAVE_PYDECK = False

# --------- Scan QR / code-barres ----------
try:
    from pyzbar.pyzbar import decode as qr_decode
    from PIL import Image
    HAVE_QR = True
except Exception:
    HAVE_QR = False

# --------- OCR (photo -> texte) ----------
try:
    import cv2, numpy as np
    from pytesseract import image_to_string
    from PIL import Image as PILImage, ImageOps
    # HEIC (optionnel) : si la lib est là on l'active, sinon on ignore
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except Exception:
        pass
    HAVE_OCR = True
except Exception:
    HAVE_OCR = False

# Carte cliquable (Folium)
try:
    import folium
    from streamlit_folium import st_folium
    HAVE_FOLIUM = True
except Exception as e:
    HAVE_FOLIUM = False
    st.error(f"Import folium/streamlit_folium impossible : {e}")






# =============================================================================
#                               CONFIG & STYLE
# =============================================================================
st.set_page_config(page_title="Copilote Livraisons", layout="wide")
st.set_option("client.showErrorDetails", True)

# --- NAV persistante (remplace st.tabs) ---
TABS = ["🧭 Guide", "📝 Saisie", "🗂️ Affectation", "📦 Exports", "⚙️ Paramètres"]
if "TAB" not in st.session_state:
    st.session_state.TAB = "🗂️ Affectation"  # démarre direct sur l'onglet utile

st.session_state.TAB = st.sidebar.radio(
    "Navigation",
    TABS,
    index=TABS.index(st.session_state.TAB),
    key="nav_tabs"
)

def tab_is(name: str) -> bool:
    return st.session_state.TAB == name

ZONES      = ["75", "91", "92", "93", "94", "95", "77", "78", "60"]
ZONE_EMOJI = {"75":"⬜️","91":"🟦","92":"🟥","93":"🟪","94":"🟫","95":"🟩","77":"🟧","78":"🟨","60":"🟣"}
ZONE_RGB   = {
    "75":[240,240,240], "91":[59,130,246], "92":[239,68,68], "93":[139,92,246],
    "94":[120,72,48],   "95":[16,185,129], "77":[251,146,60], "78":[245,158,11],
    "60":[123,63,178]
}

TOURNEES   = [str(i) for i in range(1,6)]

DEFAULT_TRUCKS = [
    {"id":"DA-045-XD","alias":"Lionel (Ancien)","couleur":"Rouge","combos":[
        {"big":13,"small":0,"label":"13 grandes"},
        {"big":10,"small":4,"label":"10 grandes + 4 petites"},
        {"big":7,"small":8,"label":"7 grandes + 8 petites"},
        {"big":0,"small":15,"label":"15 petites"}]},
    {"id":"407-FVB-92","alias":"Benjamin (Ancien)","couleur":"Rouge","combos":[
        {"big":11,"small":0,"label":"11 grandes"},
        {"big":8,"small":4,"label":"8 grandes + 4 petites"},
        {"big":5,"small":8,"label":"5 grandes + 8 petites"},
        {"big":0,"small":13,"label":"13 petites"}]},
    {"id":"BD-435-BS","alias":"Pedro (Ancien)","couleur":"Rouge","combos":[
        {"big":13,"small":0,"label":"13 grandes"},
        {"big":10,"small":4,"label":"10 grandes + 4 petites"},
        {"big":7,"small":8,"label":"7 grandes + 8 petites"},
        {"big":0,"small":15,"label":"15 petites"}]},
    {"id":"GV 587 YG","alias":"Lionel (Nouveau)","couleur":"Vert","combos":[
        {"big":11,"small":0,"label":"11 grandes"},
        {"big":8,"small":4,"label":"8 grandes + 4 petites"},
        {"big":5,"small":8,"label":"5 grandes + 8 petites"},
        {"big":0,"small":13,"label":"13 petites"}]},
    {"id":"GV 862 YG","alias":"Nouveau (Vert)","couleur":"Vert","combos":[
        {"big":11,"small":0,"label":"11 grandes"},
        {"big":8,"small":4,"label":"8 grandes + 4 petites"},
        {"big":5,"small":8,"label":"5 grandes + 8 petites"},
        {"big":0,"small":13,"label":"13 petites"}]},
]

# --- UI: thème pro ---
st.markdown("""
<style>
:root{ --bg:#0b1020; --panel:#121a33; --card:#172143;
--muted:#8aa0c0; --text:#e9f0ff; --brand:#5b8cff; --ok:#10b981; --warn:#f59e0b; --err:#ef4444; }
html, body, [data-testid="stAppViewContainer"]{background:var(--bg);}
[data-testid="stHeader"]{background:transparent;}
h1,h2,h3,h4,h5,h6, p, span, div, label{ color:var(--text)!important; }
.small{ color:var(--muted); font-size:.9rem }
.badge{ display:inline-block; padding:.12rem .45rem; border-radius:999px; border:1px solid #2b355f; background:#1a244a; color:#bcd0ff; font-weight:600; font-size:.75rem }
.card{ background:var(--card); border:1px solid #2b355f; border-radius:14px; padding:1rem; }
.callout{ background:#0e1635; border:1px solid #2b355f; border-left:3px solid var(--brand); border-radius:10px; padding:.9rem 1rem; }
.hr{ height:1px; background:#21305b; border:0; margin:1rem 0 }
.kpi{ display:flex; gap:.8rem }
.kpi .box{ background:#0e1635; border:1px solid #2b355f; border-radius:12px; padding:.7rem 1rem; min-width:120px; text-align:center }
.kpi .v{ font-weight:700; font-size:1.1rem }
.kpi .l{ color:#9db0d8; font-size:.85rem }
.help{ color:#9db0d8 }
button[kind="secondary"]{ border:1px solid #2b355f !important }
.combo{ display:inline-block; padding:.2rem .45rem; margin:.15rem; border-radius:8px; border:1px solid #2b355f; background:#101738; font-size:.8rem }
.combo.ok{ border-color:#0ea371; background:#0b1f1a }
.combo.bad{ border-color:#b45309; background:#1f1408 }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#                        PERSISTENCE (CSV) & HELPERS + CACHE
# =============================================================================
BONS_CSV = "bons.csv"

def _bons_columns():
    return [
        "id_bon","uid","date","client","num_client","telephone","ville",
        "adresse","zones","dept_affichage",
        "poids_kg","n_big","n_small","is_restaurant",
        "tournee","ordre","camion_id","zone_bucket",
        "lat","lon"
    ]

def clean_str(x) -> str:
    if x is None: return ""
    if isinstance(x, float) and math.isnan(x): return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s

@st.cache_data(show_spinner=False)
def load_bons_df_cached(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=_bons_columns())
    return pd.read_csv(path, dtype=str)

def _harmonize_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = _bons_columns()
    for c in cols:
        if c not in df.columns: df[c] = None
    df = df[cols].copy()

    # numeric conversions
    for c in ["poids_kg","n_big","n_small","ordre","lat","lon","uid"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ensure stable uid column
    if "uid" not in df.columns or df["uid"].isna().all():
        df["uid"] = list(range(1, len(df) + 1))
    else:
        existing = df["uid"].dropna().astype(int).tolist()
        next_id = max(existing) + 1 if existing else 1
        for i in df[df["uid"].isna()].index:
            df.at[i, "uid"] = next_id
            next_id += 1
        df["uid"] = df["uid"].astype(int)

    df["is_restaurant"] = df["is_restaurant"].astype(str).str.lower().isin(["true","1","yes","y","oui"])
    df["tournee"] = df["tournee"].apply(clean_str)
    df["camion_id"] = df["camion_id"].apply(clean_str)
    df["zone_bucket"] = df["zone_bucket"].apply(clean_str)
    return df

def load_bons_df() -> pd.DataFrame:
    return _harmonize_df(load_bons_df_cached(BONS_CSV))

def save_bons_df():
    if "bons" in st.session_state and isinstance(st.session_state.bons, pd.DataFrame):
        st.session_state.bons.to_csv(BONS_CSV, index=False)
        st.cache_data.clear()

def to_csv_str(df: pd.DataFrame) -> str:
    buf = StringIO(); df.to_csv(buf, index=False, encoding="utf-8"); return buf.getvalue()

def save_day_json() -> bytes:
    data = {
        "trucks": st.session_state.trucks.to_dict(orient="records"),
        "bons":   st.session_state.bons.to_dict(orient="records"),
    }
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

def load_day_json(file_bytes: bytes):
    data = json.loads(file_bytes.decode("utf-8"))
    st.session_state.trucks = pd.DataFrame(data.get("trucks", DEFAULT_TRUCKS))
    # ✅ correct: on utilise bien _harmonize_df
    st.session_state.bons = _harmonize_df(pd.DataFrame(data.get("bons", [])))
    save_bons_df()

def primary_zone(row) -> str:
    z = clean_str(row.get("dept_affichage"))
    if z: return z
    raw = clean_str(row.get("zones"))
    return raw.split(",")[0].strip() if raw else ""

def bon_chip(row) -> str:
    z = primary_zone(row); badge = ZONE_EMOJI.get(z,"⬜️")
    client = clean_str(row.get("client")) or "—"
    ville  = clean_str(row.get("ville"))  or "—"
    big    = int(row.get("n_big") or 0)
    small  = int(row.get("n_small") or 0)
    r = " • R" if bool(row.get("is_restaurant")) else ""
    return f"{badge} {client} • BIG:{big} • SMALL:{small} • {ville} [{z or '?'}]{r}"

@st.cache_data(show_spinner=False)
def totals_for(df: pd.DataFrame, t: str):
    sub = df[df["tournee"].astype(str).str.strip()==t]
    return int(sub["n_big"].fillna(0).sum()), int(sub["n_small"].fillna(0).sum())

def fits_any_combo(truck_row: dict, big: int, small: int) -> bool:
    for c in truck_row.get("combos", []):
        if big <= int(c["big"]) and small <= int(c["small"]): return True
    return False


HISTORY_CSV = "history.csv"

def _history_columns():
    # même structure que bons + méta d’archive
    return _bons_columns() + ["finished_at","tournee_label","truck_id"]

@st.cache_data(show_spinner=False)
def load_history_df_cached(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=_history_columns())
    return pd.read_csv(path, dtype=str)

def load_history_df() -> pd.DataFrame:
    df = load_history_df_cached(HISTORY_CSV)
    # types numériques si besoin
    for c in ["poids_kg","n_big","n_small","ordre","lat","lon","uid"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def save_history_df(df_hist: pd.DataFrame):
    df_hist.to_csv(HISTORY_CSV, index=False)
    st.cache_data.clear()


# =============================================================================
#                 OCR ROBUSTE — Libellés + ROI ANJU + Scoring
# =============================================================================
OCR_LANGS = "fra+eng"
SUPPLIER_CP = "92390"
SUPPLIER_KEYWORDS = (
    "ANJU", "ENTREPRISE", "VILLENEUVE",
    "SCANIW", "SCANW", "SCAN I W", "SCANIWENTERPRISE"
)


# --- Prétraitements
def _pp_full(pil_img):
    import numpy as np, cv2
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    target = 2300
    scale = target / max(h, w)
    if abs(scale-1.0) > 1e-3:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g2 = clahe.apply(g)
    blur = cv2.GaussianBlur(g2, (0,0), 1.0)
    sharp = cv2.addWeighted(g2, 1.6, blur, -0.6, 0)
    _, bw = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return PILImage.fromarray(bw)

def _expand_box(box_pct, pad=2):
    x, y, w, h = box_pct
    x = max(0, x - pad); y = max(0, y - pad)
    w = min(100 - x, w + 2*pad); h = min(100 - y, h + 2*pad)
    return [x, y, w, h]

def _crop_percent(pil_img, box_pct):
    x, y, w, h = box_pct
    W, H = pil_img.size
    L = int(W * x/100.0); T = int(H * y/100.0)
    R = int(W * (x+w)/100.0); B = int(H * (y+h)/100.0)
    return pil_img.crop((L, T, R, B))

def _rotate_pil(pil_img, angle_deg):
    from PIL import Image as PILImage2
    try:
        resample = PILImage2.Resampling.BICUBIC
    except Exception:
        resample = PILImage2.BICUBIC
    return pil_img.rotate(angle_deg, expand=True, resample=resample)


def _enhance_roi(pil_roi, upscale=1.8):
    import numpy as np, cv2
    arr = np.array(pil_roi.convert("L"))
    arr = cv2.resize(arr, (int(arr.shape[1]*upscale), int(arr.shape[0]*upscale)), interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(arr, (0,0), 1.0)
    sharp = cv2.addWeighted(arr, 1.6, blur, -0.6, 0)
    _, bw = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return PILImage.fromarray(bw)

def _norm_text(s: str) -> str:
    return " ".join((s or "").replace("\u00a0", " ").split())

def _read_roi_any(pil_img, box_pct, whitelist=None):
    roi = _crop_percent(pil_img, _expand_box(box_pct, pad=2))
    return _ocr_try_many(_enhance_roi(roi), whitelist=whitelist)

def _ocr_try_many(pil_img, whitelist=None):
    texts, psms = [], [7,6,4,3]
    for v in [pil_img, _enhance_roi(pil_img, 1.5)]:
        for p in psms:
            cfg = f"--oem 1 --psm {p}"
            if whitelist: cfg += f" -c tessedit_char_whitelist={whitelist}"
            try:
                t = image_to_string(v, lang=OCR_LANGS, config=cfg).strip()
            except Exception:
                t = image_to_string(v, config=cfg).strip()
            if t: texts.append(t)
    return max(texts, key=len) if texts else ""

# --- Utilitaires parsing
def _rx(text, patterns, flags=0, group=1):
    for p in patterns:
        m = re.search(p, text, flags)
        if m: return m.group(group).strip()
    return ""

def _parse_date_fr(s: str) -> str:
    if not s: return ""
    s = s.replace(" ", "")
    m = re.search(r"(\d{1,2})[\/\.\-](\d{1,2})[\/\.\-](\d{2,4})", s)
    if not m: return ""
    d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y < 100: y += 2000
    from datetime import date as _date
    try: return _date(y, mth, d).isoformat()
    except: return ""

def _title_clean(s: str) -> str:
    return " ".join((s or "").split()).title()

# --- 1) Lecture PLEIN-TEXTE par libellés tolérants
LABEL_PATTERNS = {
  "id_bon": [
    r"B[O0]N\s*DE\s*C[CDO0]E\s*(?:N[°Oº]|NO|Nº|N°)\s*[:\-]?\s*([A-Z0-9\/\.\- ]{2,})",
    r"B[O0]N\s*DE\s*COMMAN[DEO]\s*(?:N[°Oº]|NO|Nº|N°)\s*[:\-]?\s*([A-Z0-9\/\.\- ]{2,})",
  ],
  "date": [
    r"D[AÂ]TE?\s*[:\-]?\s*([0-9./\- ]{6,10})",
    r"Date\s*[:\-]?\s*([0-9./\- ]{6,10})",
  ],
  "num_client": [
    r"(?:C[O0]DE\s*CLIENT|CODECLIENT|Code\s*client|Codeclient)\s*[:\-]?\s*([A-Z0-9 ]{4,})",
  ],
  "telephone": [
    r"(?:T[ée]l[ée]phone|Telephone|T[ée]l\.?|Tel\.?)\s*[:\-]?\s*([\d ./+()-]{8,})",
  ],
  "poids": [
    r"\bP[O0]IDS\b\s*[:\-]?\s*([0-9][0-9\s\.,]{0,10})",
    r"\bP[O0]IDS\b.*?([0-9][0-9\s\.,]{0,10})",
  ],
  "cp_ville": [
    r"\b(\d{5})\s+([A-ZÉÈÀÙÂÊÎÔÛÇ' \-]{3,})",
  ],
  "nb_pal": [
    r"NBRE\s*DE\s*PAL[^\d]{0,6}(\d{1,3})",
  ],
  "tva_cee": [
    r"TVA\s*(?:INTRA\w*|CEE)\s*[:\-]?\s*([A-Z0-9\s]{5,})",
  ],
  "total_marchandises": [
    r"TOTAL\s+MARCHANDISES\s*[:\-]?\s*([0-9\s\.,]+)",
  ],
  "tva_total": [
    r"(?:TOTAL\s+T\.?V\.?A\.?|T\.?V\.?A\.?\s+TOTAL)\s*[:\-]?\s*([0-9\s\.,]+)",
    r"TVA\s+\(.*?\)\s*[:\-]?\s*([0-9\s\.,]+)",
  ],
  "total_payable": [
    r"TOTAL\s+(?:A\s+PAYER|PAYABLE|A\s+PAYER\s+TTC)\s*[:\-]?\s*([0-9\s\.,]+)",
    r"NET\s+A\s+PAYER\s*[:\-]?\s*([0-9\s\.,]+)",
  ],
}

def _extract_by_labels(pil_img, *, read_pallets=False):
    proc = _pp_full(pil_img)
    try:
        T = image_to_string(proc, lang=OCR_LANGS, config="--oem 1 --psm 3")
    except Exception:
        T = image_to_string(proc, config="--oem 1 --psm 3")
    T = T.replace("\u00a0", " ")
    Tu = T.upper()

    id_bon = _rx(Tu, LABEL_PATTERNS["id_bon"]).strip(" :.-").replace(" ", "")
    date_iso = _parse_date_fr(_rx(T, LABEL_PATTERNS["date"]))
    num_client = re.sub(r"\s+", "", _rx(T, LABEL_PATTERNS["num_client"]))
    telephone  = re.sub(r"[^\d+]", "", _rx(T, LABEL_PATTERNS["telephone"]))

    cp_ville_line = _rx(Tu, LABEL_PATTERNS["cp_ville"], flags=re.MULTILINE, group=0)
    cp, ville = "", ""
    if cp_ville_line:
        m = re.search(LABEL_PATTERNS["cp_ville"][0], cp_ville_line)
        if m: cp, ville = m.group(1), _title_clean(m.group(2).strip())
    # Filtre fournisseur si on est clairement sur le bloc du fournisseur
    is_supplier_block = (cp == SUPPLIER_CP) and any(k in Tu for k in SUPPLIER_KEYWORDS)
    if is_supplier_block:
        cp = ""
        ville = ""

    client, adresse = "", ""
    if cp:
        lines = [ln for ln in T.splitlines() if ln.strip()]
        idx = next((i for i, ln in enumerate(lines) if cp in ln), None)
        if idx is not None:
            up = lines[max(0, idx-4):idx]
            if up:
                adresse = ", ".join([x.strip() for x in up[-2:]])
                if len(up) >= 3:
                    cand = up[-3].strip()
                    if len(cand) >= 3 and "FRANCE" not in cand.upper() and "TVA" not in cand.upper():
                        client = _title_clean(cand)

    # Si le texte ressemble à celui du fournisseur ET que le CP est celui du fournisseur,
# on neutralise le destinataire pour éviter la confusion
    looks_like_supplier_dest = any(k in (client or "").upper() for k in SUPPLIER_KEYWORDS) \
                            or any(k in (adresse or "").upper() for k in SUPPLIER_KEYWORDS)
    if (cp == SUPPLIER_CP) and looks_like_supplier_dest:
        client = ""
        adresse = ""


        # Si ça sent le fournisseur, on neutralise
    if any(k in (client or "").upper() for k in SUPPLIER_KEYWORDS) or any(k in (adresse or "").upper() for k in SUPPLIER_KEYWORDS):
        client, adresse = "", ""

    dept = cp[:2] if len(cp) >= 2 else ""
    zones = [dept] if dept in ZONES else []


    poids_txt = _rx(T, LABEL_PATTERNS["poids"], flags=re.IGNORECASE)
    poids_kg = None
    if poids_txt:
        try:
            poids_kg = float(poids_txt.replace(" ", "").replace(",", "."))
        except Exception:
            poids_kg = None

    tva_cee = _rx(Tu, LABEL_PATTERNS.get("tva_cee", []))
    total_march = _rx(T, LABEL_PATTERNS.get("total_marchandises", []), flags=re.IGNORECASE)
    tva_total = _rx(T, LABEL_PATTERNS.get("tva_total", []), flags=re.IGNORECASE)
    total_payable = _rx(T, LABEL_PATTERNS.get("total_payable", []), flags=re.IGNORECASE)

    n_big = 0; n_small = 0
    if read_pallets:
        nbp = _rx(Tu, LABEL_PATTERNS["nb_pal"])
        try: n_big = int(nbp) if nbp else 0
        except: n_big = 0

    return {
        "id_bon": id_bon, "date": date_iso,
        "client": _title_clean(client), "num_client": num_client, "telephone": telephone,
        "ville": _title_clean(ville), "adresse": " ".join(adresse.split()),
        "cp": cp,
        "zones": zones, "dept_affichage": dept, "poids_kg": poids_kg,
        "tva_cee": clean_str(tva_cee),
        "total_marchandises": total_march,
        "tva_total": tva_total,
        "total_payable": total_payable,
        "n_big": n_big, "n_small": n_small, "is_restaurant": False,
        "_raw_text": T
    }

# --- 2) ROI calibrées ANJU (portrait A4) — coordonnées en % sur la page complète
ANJU_TPL = {
    # Haut-gauche (infos bon)
    "id_bon"    : [10, 18, 35, 6],   # "BON DE CDE N° 03908"
    "date"      : [10, 24, 22, 6],   # "Date : 05/09/25"
    "num_client": [10, 30, 22, 6],   # "Code client : ..."
    "telephone" : [10, 36, 22, 6],   # "Téléphone : ..."

    # Haut-droite (bloc destinataire — ce qu'il FAUT capter)
    "client"    : [58, 17, 34, 6],   # "JETT FOOD"
    "adresse"   : [58, 23, 34, 10],  # "75 RUE RATEAU"
    "ville_cp"  : [58, 30, 34, 7],   # "93120 LA COURNEUVE"  (on parse CP + Ville)
    # (pays/tva éventuellement sous la ville, pas nécessaire ici)

    # Bas-droite (tableau totaux)
    "poids"     : [72, 86, 15, 8],   # case "Poids" (ex: "418,04")
}

def _read_right_header_block(pil_img):
    """
    Lis le bloc destinataire en haut-droite (ANJU portrait A4).
    - Essaie plusieurs boîtes + rotations (0/90/270/180)
    - Rejette tout ce qui ressemble au fournisseur (92390, ANJU, VILLENEUVE...)
    """
    CAND_BOXES = [
        [55, 14, 40, 24],   # large (ta photo)
        [56, 15, 38, 22],   # variante
        [58, 17, 34, 18],   # plus petit
    ]
    ANGLES = [0, 90, 270, 180]  # certaines photos sont prises en travers

    for ang in ANGLES:
        img = _rotate_pil(pil_img, ang) if ang else pil_img
        for BOX in CAND_BOXES:
            roi = _crop_percent(img, BOX)
            roi = _enhance_roi(roi, upscale=2.0)
            txt = _ocr_try_many(roi)
            T = _norm_text(txt).upper()

            if not T:
                continue

            # hard-filter fournisseur
            if (SUPPLIER_CP in T) or any(k in T for k in SUPPLIER_KEYWORDS):
                continue

            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if not lines:
                continue

            client  = _title_clean(lines[0])
            adresse = _norm_text(lines[1] if len(lines) >= 2 else "")
            city    = _norm_text(lines[2] if len(lines) >= 3 else "")

            # Peut être sur 2e ligne
            m = re.search(r"(\d{5})\s+([A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜ' \-]{3,})", (city or "").upper())
            if not m:
                for i in range(1, min(4, len(lines))):
                    m = re.search(r"(\d{5})\s+([A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜ' \-]{3,})", lines[i].upper())
                    if m:
                        city = lines[i]
                        adresse = _norm_text(" ".join(lines[1:i]))
                        break

            if not m:
                continue

            cp = m.group(1)
            ville = _title_clean(m.group(2))

            # garde uniquement un CP client ≠ fournisseur
            if (cp == SUPPLIER_CP) or "VILLENEUVE" in ville.upper():
                continue

            out = {
                "client": client,
                "adresse": adresse,
                "cp": cp,
                "ville": ville,
                "dept_affichage": cp[:2] if len(cp) >= 2 else "",
                "zones": [cp[:2]] if len(cp) >= 2 and cp[:2] in ZONES else [],
            }
            return out

    return {}  # rien trouvé

def _extract_by_roi(pil_img, second_page_for_weight=None, pad_pct=2, angle_deg=0):
    """
    Ordre strict pour ANJU en portrait A4 :
      0) Bloc destinataire fixe haut-droite (client / adresse / CP+ville)
      1) Si raté : ROI dynamique autour du CP le plus à droite (≠ fournisseur)
      2) Si raté : anciennes ROI fixes + lecture par libellés
      3) Poids en bas-droite (page 1 ou 2)
    """
    out = {}

    # ---------- (0) Bloc haut-droite : destinataire ----------
    out0 = _read_right_header_block(pil_img)
    got_dest = bool(out0)
    if got_dest:
        out.update(out0)

    # ---------- (1) ROI dynamique autour d'un CP à droite ----------
    if not got_dest:
        dyn_roi, _ = _find_rightmost_cp_roi(pil_img)
        if dyn_roi is not None:
            txt = _ocr_try_many(_enhance_roi(dyn_roi))
            T = _norm_text(txt).upper()

            # CP + Ville
            cp, ville = "", ""
            m = re.search(r"(\d{5})\s+([A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜ' \-]{3,})", T)
            if m:
                cp, ville = m.group(1), _title_clean(m.group(2))
            else:
                m2 = re.search(r"(\d{5})", T)
                if m2:
                    cp = m2.group(1)
                    after = T.split(cp, 1)[-1].strip(" -")
                    ville = _title_clean(after.splitlines()[0] if after else "")

            # Lignes -> client + adresse (avant la ligne CP)
            lines = [x.strip() for x in txt.splitlines() if x.strip()]
            client = ""; adresse = ""
            if lines:
                bad = ("ANJU", "ENTREPRISE", "BON DE CDE", "FAX", "TEL")
                usable = [l for l in lines if not any(b in l.upper() for b in bad)]
                if usable:
                    client = _title_clean(usable[0])
                    adr_parts = []
                    for l in usable[1:]:
                        if re.search(r"\d{5}", l): break
                        adr_parts.append(l)
                    adresse = _norm_text(" ".join(adr_parts))

            if cp and (cp != SUPPLIER_CP):
                out.update({
                    "cp": cp,
                    "ville": ville,
                    "dept_affichage": cp[:2],
                    "zones": [cp[:2]] if cp[:2] in ZONES else [],
                    "client": client,
                    "adresse": adresse
                })
                got_dest = True

    # ---------- (2) ROI fixes ANJU (secours) ----------
    if not got_dest:
        raw_id = _read_roi_any(pil_img, ANJU_TPL["id_bon"],
                               whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/:.- ")
        m = re.search(r"(?:BON\s*DE\s*CDE\s*(?:N[°Oº]|NO|Nº|N°)\s*[:\-]?\s*)?([A-Z0-9\/\.\-]{2,})",
                      (raw_id or "").upper())
        out["id_bon"] = (m.group(1) if m else raw_id).strip(" :.-").replace(" ", "")

        out["date"] = _parse_date_fr(_read_roi_any(
            pil_img, ANJU_TPL["date"], whitelist="0123456789/-. "
        ))

        out["num_client"] = re.sub(r"\s+", "", _read_roi_any(
            pil_img, ANJU_TPL["num_client"], whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ))

        tel = _read_roi_any(pil_img, ANJU_TPL["telephone"], whitelist="0123456789+ ()-.")
        out["telephone"] = re.sub(r"[^\d+]", "", tel or "")

        client_fix = _read_roi_any(
            pil_img, ANJU_TPL["client"],
            whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜ' -/&."
        )
        adr_fix = _read_roi_any(
            pil_img, ANJU_TPL["adresse"],
            whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜ0123456789,'-/ ."
        )
        vline = _read_roi_any(
            pil_img, ANJU_TPL["ville_cp"],
            whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜ0123456789' -"
        )
        vlineU = _norm_text(vline).upper()
        mm = re.search(r"(\d{5})\s+([A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜ' \-]{3,})", vlineU)

        if mm:
            cp = mm.group(1); ville = _title_clean(mm.group(2))
            if cp != SUPPLIER_CP:
                out.update({
                    "client": _title_clean(_norm_text(client_fix)),
                    "adresse": _norm_text(adr_fix),
                    "cp": cp,
                    "ville": ville,
                    "dept_affichage": cp[:2],
                    "zones": [cp[:2]] if cp[:2] in ZONES else []
                })

    # ---------- (3) Poids ----------
    poids_txt = _read_roi_any(pil_img, ANJU_TPL["poids"], whitelist="0123456789,. ")
    if (not poids_txt) and (second_page_for_weight is not None):
        poids_txt = _read_roi_any(second_page_for_weight, ANJU_TPL["poids"], whitelist="0123456789,. ")
    try:
        out["poids_kg"] = float((poids_txt or "").replace(" ", "").replace(",", "."))
    except Exception:
        out["poids_kg"] = 0.0

    # défauts
    out.setdefault("id_bon", ""); out.setdefault("num_client", ""); out.setdefault("telephone", "")
    out.setdefault("n_big", 0); out.setdefault("n_small", 0); out.setdefault("is_restaurant", False)
    if "zones" not in out:
        z = clean_str(out.get("dept_affichage")); out["zones"] = [z] if z in ZONES else []
    return out


# --- 3) Scoring + orchestrateur
def _score_read(d: dict) -> tuple:
    if not isinstance(d, dict): return (-1,0)
    keys = ["id_bon","client","ville","adresse","dept_affichage"]
    filled = sum(1 for k in keys if clean_str(d.get(k)))
    bonus = 0
        # Rejet dur si c'est le fournisseur
    v = clean_str(d.get("ville")).upper()
    c = clean_str(d.get("cp"))
    if (c == SUPPLIER_CP) or ("VILLENEUVE" in v):
        return (-1_000_000, 0)

    if clean_str(d.get("dept_affichage")) in ZONES: bonus += 1
    if len(clean_str(d.get("telephone"))) >= 9:     bonus += 1
    if len(clean_str(d.get("num_client"))) >= 6:    bonus += 1
    try:
        if float(d.get("poids_kg") or 0) > 0:       bonus += 1
    except: pass
    length_sum = sum(len(clean_str(d.get(k,""))) for k in keys)
    return (filled + bonus, length_sum)

def extract_bon_generic(pil_img, second_page_for_weight=None, *, read_pallets=False):
    import time, os
    start = time.perf_counter()
    TIME_BUDGET = float(os.getenv("OCR_TIME_BUDGET_SEC", "5.0"))

    def _enough(d: dict) -> bool:
        if not isinstance(d, dict):
            return False
        need = ["client", "ville", "dept_affichage"]
        # id_bon/adresse peuvent être absents sur certaines photos — on reste souple
        return all(clean_str(d.get(k)) for k in need)

    def _finalize(d: dict) -> dict:
        if not isinstance(d, dict): d = {}
        d.setdefault("n_big", 0); d.setdefault("n_small", 0); d.setdefault("is_restaurant", False)
        if "zones" not in d:
            z = clean_str(d.get("dept_affichage")); d["zones"] = [z] if z in ZONES else []
        return d

    best, best_sc = None, (-1, 0)

    # ---- 1) PRIORITÉ ROI (moitié droite / ignore 92390)
    angle_sets = [[0], [-2, 2], [-4, 4]]
    PAD_LIST = [2, 3, 1]

    stop = False
    for angles in angle_sets:
        if stop or (time.perf_counter() - start) > TIME_BUDGET: break
        for ang in angles:
            if stop or (time.perf_counter() - start) > TIME_BUDGET: break
            for pad in PAD_LIST:
                if (time.perf_counter() - start) > TIME_BUDGET: 
                    stop = True; break
                try:
                    cand = _extract_by_roi(pil_img, second_page_for_weight, pad_pct=pad, angle_deg=ang)
                    sc = _score_read(cand)
                    if sc > best_sc:
                        best, best_sc = cand, sc
                    if _enough(cand):
                        return _finalize(cand)
                except Exception:
                    continue

    # ---- 2) SECOURS : lecture par libellés (plein-texte)
    try:
        cand = _extract_by_labels(pil_img, read_pallets=read_pallets)
        # garde uniquement si la “ville” est cohérente côté destinataire
        # (si on retombe sur 92390, ça restera moins bon et n’écrasera pas un ROI correct)
        sc = _score_read(cand)
        if sc > best_sc:
            best, best_sc = cand, sc
    except Exception:
        pass

    return _finalize(best)


import pytesseract as _pyt

def _find_rightmost_cp_roi(pil_img, expand_px=60):
    import pytesseract as _pyt
    try:
        enh = _pp_full(pil_img).convert("RGB")
        data = _pyt.image_to_data(enh, lang=OCR_LANGS, config="--oem 1 --psm 6", output_type=_pyt.Output.DICT)
    except Exception:
        return None, None

    boxes = []
    for i, txt in enumerate(data.get("text", [])):
        t = (txt or "").strip()
        if re.fullmatch(r"\d{5}", t):
            boxes.append({
                "cp": t,
                "x": int(data["left"][i]), "y": int(data["top"][i]),
                "w": int(data["width"][i]), "h": int(data["height"][i]),
            })
    if not boxes: 
        return None, None

    W = pil_img.width
    right = [b for b in boxes if (b["x"] + b["w"]/2) > 0.55*W]
    right_ok = [b for b in right if b["cp"] != "92390"]
    any_ok   = [b for b in boxes if b["cp"] != "92390"]

    cand = right_ok or right or any_ok or boxes
    b = max(cand, key=lambda z: z["x"])
    L = max(0, b["x"] - expand_px)
    T = max(0, b["y"] - 3*expand_px)
    R = min(pil_img.width,  b["x"] + b["w"] + 10*expand_px)
    B = min(pil_img.height, b["y"] + b["h"] + 4*expand_px)
    return pil_img.crop((L, T, R, B)), (L, T, R, B)


# =============================================================================
#                                    STATE
# =============================================================================
if "trucks" not in st.session_state:
    st.session_state.trucks = pd.DataFrame(DEFAULT_TRUCKS)

if "bons" not in st.session_state:
    st.session_state.bons = load_bons_df()

if "tournee_truck" not in st.session_state:
    st.session_state.tournee_truck = {}

if "add_date" not in st.session_state:
    st.session_state["add_date"] = date.today()

if HAVE_GEOPY and "geocoder" not in st.session_state:
    st.session_state.geocoder = Nominatim(user_agent="copilote_livraisons")
    # Delai 1s pour respecter Nominatim. Tu peux baisser à 0.5 si nécessaire.
    st.session_state.rate = RateLimiter(st.session_state.geocoder.geocode, min_delay_seconds=1)

# Plusieurs tracés en parallèle : { "1":[uid,...], "2":[...], ... }
if "route_sels" not in st.session_state:
    st.session_state.route_sels = {t: [] for t in TOURNEES}

# Tournée actuellement éditée (celle qui reçoit les clics)
if "active_route_tour" not in st.session_state:
    st.session_state.active_route_tour = TOURNEES[0]  # par défaut "1"




# =============================================================================
#                       FONCTIONS DE GEOCODAGE (auto & manuel)
# =============================================================================
def _compose_address_for_geocode(adresse: str, ville: str) -> str:
    base = " ".join([clean_str(adresse), clean_str(ville)]).strip()
    return base

def geocode_address(addr: str):
    if not HAVE_GEOPY: return None
    try:
        return st.session_state.rate(addr) if addr else None
    except Exception:
        return None

def geocode_row_into_df(df: pd.DataFrame, idx: int, addr_override: str = "") -> bool:
    """Géocode la ligne idx du DF: utilise addr_override si fourni, sinon adresse+ville."""
    if not HAVE_GEOPY: return False
    r = df.loc[idx]
    addr = clean_str(addr_override) or _compose_address_for_geocode(r.get("adresse"), r.get("ville"))
    if not addr: return False
    loc = geocode_address(addr)
    if loc:
        df.at[idx, "lat"] = float(loc.latitude)
        df.at[idx, "lon"] = float(loc.longitude)
        return True
    return False


# =============================================================================
#        Helpers "Patron" — plan de tournées + polylines (carte existante)
# =============================================================================
DEPOT = {"lat": 48.907, "lon": 2.438}     # Point de départ/retour
# --- Réglages demandés
DEPOT_ADDR = "10 Av. Marcellin Berthelot, 92390 Villeneuve-la-Garenne, France"

# Ordre de remplissage des camions
TRUCK_ORDER = ["DA-045-XD", "407-FVB-92", "BD-435-BS", "GV 587 YG", "GV 862 YG"]

def _ensure_depot():
    """Essaie de géocoder l'adresse du dépôt pour surcharger DEPOT (optionnel)."""
    if HAVE_GEOPY and "rate" in st.session_state:
        try:
            loc = st.session_state.rate(DEPOT_ADDR)
            if loc:
                DEPOT["lat"] = float(loc.latitude)
                DEPOT["lon"] = float(loc.longitude)
        except Exception:
            pass

def _dist_depot(lat, lon) -> float:
    return _hv_km(float(lat or 0), float(lon or 0), DEPOT["lat"], DEPOT["lon"])

def _fits_combo(loadB, loadS, addB, addS, capB, capS) -> bool:
    return (loadB + addB) <= capB and (loadS + addS) <= capS

PATRON_TOURS_MIN = 4
PATRON_TOURS_MAX = 5
PATRON_W = {
    "capacity_exceed": 1_000_000,
    "south_north_break": 300,
    "dist_delta": 6.0,
    "zone_mix_other": 500,
    "zone_75_to_93": -50,
    "rest_not_first": 100,
}

def _primary_zone_patron(r):
    d = clean_str(r.get("dept_affichage"))
    if d: return d
    z = clean_str(r.get("zones"))
    return z.split(",")[0].strip() if z else ""

def _hv_km(a_lat,a_lon,b_lat,b_lon):
    if any(pd.isna([a_lat,a_lon,b_lat,b_lon])): return 0.0
    R=6371.0
    p1,p2=math.radians(a_lat),math.radians(b_lat)
    dphi=math.radians(b_lat-a_lat); dlbd=math.radians(b_lon-a_lon)
    x=math.sin(dphi/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dlbd/2)**2
    return 2*R*math.asin(math.sqrt(x))

def _route_len_with_return(df, idxs):
    if not idxs: return 0.0
    d = 0.0
    first = df.loc[idxs[0]]
    last  = df.loc[idxs[-1]]

    # Aller : dépôt -> premier point
    d += _hv_km(DEPOT["lat"], DEPOT["lon"], float(first["lat"] or 0), float(first["lon"] or 0))

    # Chaînage entre points
    for a, b in zip(idxs, idxs[1:]):
        ra, rb = df.loc[a], df.loc[b]
        d += _hv_km(float(ra["lat"] or 0), float(ra["lon"] or 0),
                    float(rb["lat"] or 0), float(rb["lon"] or 0))

    # Retour : dernier point -> dépôt
    d += _hv_km(float(last["lat"] or 0), float(last["lon"] or 0), DEPOT["lat"], DEPOT["lon"])
    return d


def _south_north_break(df, order_idx, pos, new_row):
    latN=float(new_row.get("lat") or 0)
    prev_i=order_idx[pos-1] if pos>0 else None
    next_i=order_idx[pos] if pos<len(order_idx) else None
    ok=True
    if prev_i is not None and latN < float(df.loc[prev_i].get("lat") or 0) - 1e-4: ok=False
    if next_i is not None and latN > float(df.loc[next_i].get("lat") or 0) + 1e-4: ok=False
    return 0 if ok else 1

def _best_combo(tr):
    best=(0,0)
    for c in tr.get("combos", []):
        t=(int(c.get("big",0)), int(c.get("small",0)))
        if t>best: best=t
    return best

def _tour_caps(trucks_df, truck_id):
    if not truck_id: return (999,999)
    row=trucks_df[trucks_df["id"].astype(str)==str(truck_id)]
    if row.empty: return (999,999)
    return _best_combo(row.iloc[0].to_dict())

def _tour_zone_dom(df, idxs):
    if not idxs: return ""
    z=df.loc[idxs].apply(_primary_zone_patron, axis=1).value_counts()
    return z.index[0] if len(z)>0 else ""

def _ensure_min_tours_labels(df, desired):
    active=sorted({clean_str(t) for t in df["tournee"].astype(str) if clean_str(t)})
    missing=max(0, desired-len(active))
    return active, missing

def _new_tour_label(df):
    existing={clean_str(x) for x in df["tournee"].astype(str).unique()}
    for k in range(1,10):
        if str(k) not in existing: return str(k)
    return "9"

def _southernmost_unassigned(df):
    pool=df[(df["tournee"].apply(clean_str)=="") & df["lat"].notna() & df["lon"].notna()]
    if pool.empty: return None
    return pool.sort_values("lat", ascending=True).index[0]

def _restaurants_first(df, idxs):
    if not idxs: return []
    df_t=df.loc[idxs]
    rest=df_t[df_t["is_restaurant"]==True].index.tolist()
    other=df_t[df_t["is_restaurant"]!=True].index.tolist()
    return rest+other

def _zone_mix_penalty(zone_before, zone_new):
    if not zone_before or not zone_new: return 0
    if zone_before==zone_new: return 0
    if zone_before=="75" and zone_new=="93": return PATRON_W["zone_75_to_93"]
    return PATRON_W["zone_mix_other"]

def _score_insertion(df, order_idx, pos, new_row, truck_id, zone_dom):
    capB,capS=_tour_caps(st.session_state.trucks, truck_id)
    addB,addS=int(new_row["n_big"] or 0), int(new_row["n_small"] or 0)
    loadB=int(df.loc[order_idx]["n_big"].fillna(0).sum()) if order_idx else 0
    loadS=int(df.loc[order_idx]["n_small"].fillna(0).sum()) if order_idx else 0
    if loadB+addB>capB or loadS+addS>capS: return PATRON_W["capacity_exceed"]

    score = PATRON_W["south_north_break"] * _south_north_break(df, order_idx, pos, new_row)

    old_len=_route_len_with_return(df, order_idx)
    new_idxs=order_idx.copy(); new_idxs.insert(pos, int(new_row.name))
    new_idxs=_restaurants_first(df, new_idxs)
    new_len=_route_len_with_return(df, new_idxs)
    score += PATRON_W["dist_delta"] * max(0.0, new_len-old_len)

    score += _zone_mix_penalty(zone_dom, _primary_zone_patron(new_row))
    if bool(new_row["is_restaurant"]) and pos!=0:
        score += PATRON_W["rest_not_first"]
    return score

def patron_build_plan(df_src: pd.DataFrame) -> dict:
    """
    Remplit les camions l'un après l'autre dans TRUCK_ORDER.
    - Priorité restaurants : on démarre par le resto le plus loin du dépôt, puis on ajoute
      uniquement des restos tant que possible (ils restent en tête de tournée).
    - Sinon, seed = point le plus loin du dépôt.
    - Ajout glouton par insertion : à chaque étape, on choisit le bon admissible (respect combos)
      qui minimise le surcoût de distance (Haversine avec retour dépôt). Égalité → plus lourd.
    - Poids non contraignant (juste tie-breaker et affichage).
    - Recalcule sur tous les bons non envoyés (camion_id vide), coords requises.
    """
    df = df_src.copy()

    # Bons éligibles = non envoyés (camion_id vide) + géocodés
    elig = df[(df["camion_id"].apply(clean_str) == "") & df["lat"].notna() & df["lon"].notna()]
    remaining = set(elig.index.tolist())

    tours = []
    tcount = 0

    for tid in TRUCK_ORDER:
        if not remaining: break

        capB, capS = _tour_caps(st.session_state.trucks, tid)
        order = []              # liste d'index bons
        rest_block_len = 0      # longueur du bloc "restaurants" en tête
        loadB = 0; loadS = 0

        # ---------- Phase 1 : restaurants en priorité ----------
        def fits_idx(i: int) -> bool:
            addB = int(df.at[i, "n_big"] or 0)
            addS = int(df.at[i, "n_small"] or 0)
            return _fits_combo(loadB, loadS, addB, addS, capB, capS)

        # seed resto = le plus loin du dépôt parmi ceux qui tiennent
        rest_pool = [i for i in list(remaining) if bool(df.at[i, "is_restaurant"]) and fits_idx(i)]
        if rest_pool:
            seed = max(rest_pool, key=lambda i: _dist_depot(df.at[i, "lat"], df.at[i, "lon"]))
            order = [seed]; rest_block_len = 1
            loadB += int(df.at[seed, "n_big"] or 0); loadS += int(df.at[seed, "n_small"] or 0)
            remaining.remove(seed)

            # ajouter d'autres restos par insertion dans la zone [0 .. rest_block_len]
            while True:
                candidates = [i for i in list(remaining)
                              if bool(df.at[i, "is_restaurant"]) and fits_idx(i)]
                if not candidates: break

                base_len = _route_len_with_return(df, order)
                best = None  # (delta, -poids, i, pos)

                for i in candidates:
                    poids_i = float(df.at[i, "poids_kg"] or 0)
                    for pos in range(0, rest_block_len + 1):
                        new_order = order.copy(); new_order.insert(pos, i)
                        delta = _route_len_with_return(df, new_order) - base_len
                        key = (delta, -poids_i, i, pos)
                        if best is None or key < best[0]:
                            best = (key, i, pos)

                if best is None: break
                (_, i_best, pos_best) = best
                order.insert(pos_best, i_best)
                rest_block_len += 1
                loadB += int(df.at[i_best, "n_big"] or 0); loadS += int(df.at[i_best, "n_small"] or 0)
                remaining.remove(i_best)

        # ---------- Phase 2 : si camion encore vide, seed "farthest first" ----------
        if not order:
            candidates = [i for i in list(remaining) if fits_idx(i)]
            if candidates:
                seed = max(candidates, key=lambda i: _dist_depot(df.at[i, "lat"], df.at[i, "lon"]))
                order = [seed]
                loadB += int(df.at[seed, "n_big"] or 0); loadS += int(df.at[seed, "n_small"] or 0)
                remaining.remove(seed)

        # ---------- Phase 3 : compléter avec non-restaurants ----------
        while order:
            candidates = [i for i in list(remaining)
                          if (not bool(df.at[i, "is_restaurant"])) and fits_idx(i)]
            if not candidates: break

            base_len = _route_len_with_return(df, order)
            best = None  # (delta, -poids, i, pos)

            for i in candidates:
                poids_i = float(df.at[i, "poids_kg"] or 0)
                for pos in range(rest_block_len, len(order) + 1):
                    new_order = order.copy(); new_order.insert(pos, i)
                    delta = _route_len_with_return(df, new_order) - base_len
                    key = (delta, -poids_i, i, pos)
                    if best is None or key < best[0]:
                        best = (key, i, pos)

            if best is None: break
            (_, i_best, pos_best) = best
            order.insert(pos_best, i_best)
            loadB += int(df.at[i_best, "n_big"] or 0); loadS += int(df.at[i_best, "n_small"] or 0)
            remaining.remove(i_best)

        # Si le camion a reçu au moins un bon -> on crée une tournée
        if order:
            tcount += 1
            tours.append({
                "t": str(tcount),
                "order": order,
                "truck": tid,
                "zone": _tour_zone_dom(df, order)
            })

    return {"tours": tours}


def patron_plan_paths(df: pd.DataFrame, plan: dict, include_depot=True):
    """Construit des chemins pour PathLayer : [{'path':[[lon,lat],...],'color':[r,g,b]}]."""
    paths=[]
    for T in plan.get("tours", []):
        idxs=T["order"]
        if not idxs: continue
        coords=[]
        if include_depot: coords.append([DEPOT["lon"], DEPOT["lat"]])
        for i in idxs:
            r=df.loc[i]
            if pd.notna(r.get("lat")) and pd.notna(r.get("lon")):
                coords.append([float(r["lon"]), float(r["lat"])])
        if include_depot and coords:
            coords.append([DEPOT["lon"], DEPOT["lat"]])
        if len(coords)>=2:
            z=T.get("zone") or ""
            color=ZONE_RGB.get(z, [180,180,180])
            paths.append({"path":coords, "color":color})
    return paths

def patron_plan_annotations(df: pd.DataFrame, plan: dict):
    """Retourne un DataFrame indexé comme df avec proposed_tour / proposed_order."""
    annot = pd.DataFrame(index=df.index, columns=["proposed_tour", "proposed_order"])
    for T in plan.get("tours", []):
        order = T.get("order", []) or []
        tlab = T.get("t", "")
        for k, idx in enumerate(order, start=1):
            if idx in annot.index:
                annot.at[idx, "proposed_tour"] = str(tlab)
                annot.at[idx, "proposed_order"] = int(k)
    return annot



# =============================================================================
#                               HEADER / SIDEBAR
# =============================================================================
with st.container():
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("### **🚚 Copilote Livraisons**")
        st.markdown("<div class='small'>Saisir ➜ Glisser ➜ Camion • OCR libellés + ROI ANJU • Carte • PDF.</div>", unsafe_allow_html=True)
    with c2:
        df_head = st.session_state.bons
        st.markdown(
            f"""
            <div class="kpi">
              <div class="box"><div class="v">{len(df_head)}</div><div class="l">Bons saisis</div></div>
              <div class="box"><div class="v">{df_head['tournee'].apply(clean_str).str.len().gt(0).sum()}</div><div class="l">Affectés</div></div>
            </div>
            """,
            unsafe_allow_html=True
        )

with st.sidebar:
    st.markdown("#### 💾 Sauvegardes")
    st.download_button(
        "Télécharger la journée (.json)",
        data=save_day_json(),
        file_name=f"journee_{date.today()}.json",
        mime="application/json",
        use_container_width=True,
        key="dl_json_sidebar",
    )
    st.download_button(
        "Exporter les bons (.csv)",
        data=to_csv_str(st.session_state.bons),
        file_name=f"bons_{date.today()}.csv",
        mime="text/csv",
        use_container_width=True,
        key="dl_csv_sidebar",
    )

# -----------------------------------------------------------------------------
#                                   GUIDE
# -----------------------------------------------------------------------------
if tab_is("🧭 Guide"):
    st.markdown("### Démarrage express")
    st.markdown("""
<div class="callout">
<b>1)</b> Envoie une photo du bon : l’OCR repère les libellés et remplit les champs.<br>
<b>2)</b> Ajoute le bon — géocodage auto immédiat (si possible).<br>
<b>3)</b> La <b>carte</b> montre les points (bouton pour géocoder les manquants).<br>
<b>4)</b> Glisse un bon depuis une <b>Zone</b> vers une <b>Tournée</b> (ordre = position).<br>
<b>5)</b> Affecte la <b>Tournée</b> à un <b>Camion</b> (validation des combos).<br>
<b>6)</b> Exporte en PDF/CSV.
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
#                                   SAISIE
# -----------------------------------------------------------------------------
if tab_is("📝 Saisie"):
    # --- Caméra unique ---
    # --- Caméra / fichier
    st.markdown("### 📁 Importer une photo du bon (unique)")

    # bouton pour vider l'uploader ET changer les clés
    if st.button("🔄 Réinitialiser l'image importée"):
        st.session_state['bon_nonce'] = st.session_state.get('bon_nonce', 0) + 1
        # purge les clés internes utilisées par le composant si besoin
        for k in list(st.session_state.keys()):
            if k.startswith("bon_file_") or k.startswith("bon_camera_"):
                st.session_state.pop(k)
        st.rerun()

    nonce = st.session_state.get('bon_nonce', 0)

    # 👉 clé différente à chaque reset (important)
    pil, pil_p2 = bon_image_picker(key_prefix=f"bon_{nonce}")

    if HAVE_OCR:
        # pil/pil_p2 déjà fournis ci-dessus (bon_{nonce})
        if pil is not None:
            # OCR + extraction (réutilise ta fonction telle quelle)
            data = extract_bon_generic(pil, second_page_for_weight=pil_p2, read_pallets=False)

            with st.expander("🔍 Debug OCR (valeurs extraites)"):
                st.json({k: v for k, v in data.items() if not k.startswith("_")})

        # ... (la suite est IDENTIQUE à ton code actuel :
        # helpers _fmt_date_ddmmyyyy / _num2 / _adresse_voie,
        # puis construction de out12, st.text(out12),
        # et le pré-remplissage du formulaire dans st.session_state)
        # --> Tu peux garder ta partie existante telle quelle ici.
    else:
        st.info("OCR indisponible. Installe Tesseract + pytesseract + opencv.")

    # --- QR / code-barres (indépendant de HAVE_OCR) ---
    st.markdown("### 🔳 Scanner un QR / code-barres (option)")
    if HAVE_QR:
        qr_img = st.file_uploader(
            "Importer une image avec QR/code-barres (contient soit l'ID, soit un JSON)",
            type=["jpg","jpeg","png","heic","heif"],
            key="qr_img"
        )
        if qr_img is not None:
            try:
                qr_pil = PILImage.open(qr_img)
                results = qr_decode(qr_pil)
            except Exception:
                results = []
            if results:
                raw = results[0].data.decode("utf-8", errors="ignore").strip()
                try:
                    payload = json.loads(raw)
                except Exception:
                    payload = {"id_bon": raw}
                with st.expander("📦 Contenu du QR (debug)"):
                    st.json(payload)
                zones_payload = payload.get("zones", [])
                if isinstance(zones_payload, str):
                    zones_payload = [z.strip() for z in zones_payload.split(",") if z.strip()]
                st.session_state["add_id_bon"]   = clean_str(payload.get("id_bon"))
                st.session_state["add_client"]   = clean_str(payload.get("client"))
                st.session_state["add_num_cli"]  = clean_str(payload.get("num_client"))
                st.session_state["add_tel"]      = clean_str(payload.get("telephone"))
                st.session_state["add_ville"]    = clean_str(payload.get("ville"))
                st.session_state["add_adresse"]  = clean_str(payload.get("adresse"))
                st.session_state["add_zones"]    = [z for z in zones_payload if z in ZONES]
                st.session_state["add_poids"]    = float(payload.get("poids_kg", 0) or 0)
                st.session_state["add_n_big"]    = int(payload.get("n_big", 0) or 0)
                st.session_state["add_n_small"]  = int(payload.get("n_small", 0) or 0)
                st.session_state["add_is_rest"]  = bool(payload.get("is_restaurant", False))
                st.success("✅ QR lu. Formulaire pré-rempli.")
    else:
        st.caption("Installe `pyzbar` + `Pillow` pour activer le scan QR.")

    # --- reset programmé (AVANT de créer les widgets) ---
    if st.session_state.pop("_reset_add_bon", False):
        st.session_state.update({
            "add_id_bon": "",
            "add_client": "",
            "add_num_cli": "",
            "add_tel": "",
            "add_ville": "",
            "add_adresse": "",
            "add_zones": [],
            "add_poids": 0.0,
            "add_n_big": 0,
            "add_n_small": 0,
            "add_is_rest": False,
            "add_date": date.today(),
        })

    # ---- Formulaire d’ajout ----
    st.markdown("### Saisie d’un nouveau bon")
    with st.form("add_bon", clear_on_submit=True):
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            id_bon   = st.text_input("N° bon / référence", placeholder="FA250xxx",
                                     key="add_id_bon", value=st.session_state.get("add_id_bon",""))
            date_b = st.date_input("Date", key="add_date")
            client   = st.text_input("Client", key="add_client", value=st.session_state.get("add_client",""))
            num_cli  = st.text_input("Code client (optionnel)", key="add_num_cli", value=st.session_state.get("add_num_cli",""))
            tel      = st.text_input("Téléphone (optionnel)", key="add_tel", value=st.session_state.get("add_tel",""))
        with c2:
            ville    = st.text_input("Ville (optionnel)", key="add_ville", value=st.session_state.get("add_ville",""))
            adresse  = st.text_area("Adresse complète", key="add_adresse", value=st.session_state.get("add_adresse",""))
            zones    = st.multiselect("Zones", options=ZONES,
                                      default=st.session_state.get("add_zones", []),
                                      key="add_zones", help="Ex : 75 puis 93")
            dept     = zones[0] if zones else ""
        with c3:
            poids = st.number_input("Poids (kg)", min_value=0.0, step=10.0,
                                    key="add_poids", value=float(st.session_state.get("add_poids", 0)))
            n_big = st.number_input("Palettes BIG (100x120)", min_value=0, step=1,
                                    key="add_n_big", value=int(st.session_state.get("add_n_big", 0)))
            n_sm  = st.number_input("Palettes SMALL (80x120)", min_value=0, step=1,
                                    key="add_n_small", value=int(st.session_state.get("add_n_small", 0)))
            is_r  = st.checkbox("Restaurant (à livrer en premier)",
                                key="add_is_rest", value=bool(st.session_state.get("add_is_rest", False)))
        ok = st.form_submit_button("Ajouter le bon", use_container_width=True)

    if ok:
        df_local = st.session_state.bons
        new = {
            "id_bon": clean_str(id_bon) or f"BON-{len(df_local)+1}",
            "date": date_b.isoformat(),
            "client": clean_str(client),
            "num_client": clean_str(num_cli),
            "telephone": clean_str(tel),
            "ville": clean_str(ville),
            "adresse": clean_str(adresse),
            "zones": ",".join(zones),
            "dept_affichage": clean_str(dept),
            "poids_kg": float(poids),
            "n_big": int(n_big),
            "n_small": int(n_sm),
            "is_restaurant": bool(is_r),
            "tournee": "",
            "ordre": None,
            "camion_id": "",
            "zone_bucket": clean_str(dept) or (zones[0] if zones else ""),
            "lat": None, "lon": None,
        }

        st.session_state.bons = pd.concat([st.session_state.bons, pd.DataFrame([new])], ignore_index=True)

        # Géocodage AUTO immédiat (si geopy dispo)
        if HAVE_GEOPY:
            try:
                idx_new = st.session_state.bons.index[-1]
                addr_full = _compose_address_for_geocode(new["adresse"], new["ville"])
                loc = geocode_address(addr_full) if addr_full else None
                if loc:
                    st.session_state.bons.at[idx_new, "lat"] = float(loc.latitude)
                    st.session_state.bons.at[idx_new, "lon"] = float(loc.longitude)
                    st.toast("📍 Adresse géocodée automatiquement", icon="✅")
                else:
                    st.toast("📍 Géocodage auto indisponible pour ce bon (tu peux tenter via les boutons).", icon="⚠️")
            except Exception:
                st.toast("📍 Erreur géocodage auto — tente via les boutons.", icon="⚠️")

        save_bons_df()
        st.session_state["_reset_add_bon"] = True
        st.rerun()

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("#### Aperçu des bons saisis")
    if st.session_state.bons.empty:
        st.info("Aucun bon pour l’instant.")
    else:
        cols = ["id_bon","client","ville","zones","poids_kg","n_big","n_small","tournee","camion_id","lat","lon"]
        show = st.session_state.bons[cols].copy().sort_values(by=["zones","id_bon"])
        st.dataframe(show.head(300), use_container_width=True, hide_index=True, height=360)
        st.caption(f"{len(show)} lignes au total")

    # --- ÉDITION / SUPPRESSION ---
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### 🛠 Modifier / Supprimer un bon (avec géocodage manuel)")

    df_edit = st.session_state.bons
    if df_edit.empty:
        st.info("Aucun bon à modifier pour l’instant.")
    else:
        ids = df_edit["id_bon"].astype(str).tolist()
        sel_id = st.selectbox("Sélectionne un bon", ids, index=0, key="edit_sel_id")

        idx_list = df_edit.index[df_edit["id_bon"].astype(str) == sel_id].tolist()
        if not idx_list:
            st.warning("Bon introuvable.")
        else:
            idx = idx_list[0]
            row = df_edit.loc[idx]
            with st.form("form_edit_bon"):
                c1, c2, c3 = st.columns([1,1,1])
                with c1:
                    e_client  = st.text_input("Client", value=clean_str(row.get("client")))
                    e_num_cli = st.text_input("Code client", value=clean_str(row.get("num_client")))
                    e_tel     = st.text_input("Téléphone", value=clean_str(row.get("telephone")))
                    e_ville   = st.text_input("Ville", value=clean_str(row.get("ville")))
                with c2:
                    e_adresse = st.text_area("Adresse complète", value=clean_str(row.get("adresse")))
                    cur_zones = [z.strip() for z in clean_str(row.get("zones")).split(",") if z.strip()]
                    e_zones   = st.multiselect("Zones", options=ZONES, default=cur_zones)
                    e_dept    = st.text_input(
                        "Dept affichage (1er code zone conseillé)",
                        value=clean_str(row.get("dept_affichage")) or (cur_zones[0] if cur_zones else "")
                    )
                with c3:
                    e_poids = st.number_input("Poids (kg)", min_value=0.0, step=10.0, value=float(row.get("poids_kg") or 0))
                    e_big   = st.number_input("Palettes BIG", min_value=0, step=1, value=int(row.get("n_big") or 0))
                    e_small = st.number_input("Palettes SMALL", min_value=0, step=1, value=int(row.get("n_small") or 0))
                    e_rest  = st.checkbox("Restaurant", value=bool(row.get("is_restaurant")))

                # Bloc géocodage manuel
                st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
                st.markdown("#### 📍 Géocodage manuel pour ce bon")
                cga, cgb, cgc = st.columns([2,1,1])
                with cga:
                    addr_try = st.text_input(
                        "Adresse de géocodage (laisser vide pour utiliser Adresse + Ville ci-dessus)",
                        value=_compose_address_for_geocode(row.get("adresse",""), row.get("ville","")),
                        key="edit_geo_addr"
                    )
                with cgb:
                    lat_manual = st.text_input("Latitude (optionnel)", value=str(row.get("lat") if pd.notna(row.get("lat")) else ""))
                with cgc:
                    lon_manual = st.text_input("Longitude (optionnel)", value=str(row.get("lon") if pd.notna(row.get("lon")) else ""))

                c4, c5, c6 = st.columns([1,1,1])
                save_btn = c4.form_submit_button("💾 Enregistrer", use_container_width=True)
                geo_btn  = c5.form_submit_button("📍 Géocoder cette adresse", use_container_width=True)
                del_btn  = c6.form_submit_button("🗑️ Supprimer", use_container_width=True)

                if save_btn:
                    st.session_state.bons.at[idx, "client"]          = clean_str(e_client)
                    st.session_state.bons.at[idx, "num_client"]      = clean_str(e_num_cli)
                    st.session_state.bons.at[idx, "telephone"]       = clean_str(e_tel)
                    st.session_state.bons.at[idx, "ville"]           = clean_str(e_ville)
                    st.session_state.bons.at[idx, "adresse"]         = clean_str(e_adresse)
                    st.session_state.bons.at[idx, "zones"]           = ",".join(e_zones)
                    st.session_state.bons.at[idx, "dept_affichage"]  = clean_str(e_dept)
                    st.session_state.bons.at[idx, "poids_kg"]        = float(e_poids)
                    st.session_state.bons.at[idx, "n_big"]           = int(e_big)
                    st.session_state.bons.at[idx, "n_small"]         = int(e_small)
                   
                    st.session_state.bons.at[idx, "is_restaurant"]   = bool(e_rest)

                    # Si lat/lon manuels saisis, on les applique
                    lat_val = clean_str(lat_manual); lon_val = clean_str(lon_manual)
                    if lat_val and lon_val:
                        try:
                            st.session_state.bons.at[idx, "lat"] = float(lat_val)
                            st.session_state.bons.at[idx, "lon"] = float(lon_val)
                            st.toast("📍 Coordonnées manuelles enregistrées", icon="✅")
                        except Exception:
                            st.toast("⚠️ Coordonnées manuelles invalides", icon="⚠️")

                    save_bons_df(); st.success("✅ Bon mis à jour"); st.rerun()

                if geo_btn:
                    if not HAVE_GEOPY:
                        st.warning("Installe `geopy` :  py -m pip install geopy")
                    else:
                        ok_geo = geocode_row_into_df(st.session_state.bons, idx, addr_override=addr_try)
                        if ok_geo:
                            save_bons_df(); st.success("✅ Adresse géocodée"); st.rerun()
                        else:
                            st.error("Adresse non trouvée. Essaye une variante (numéro, code postal, pays).")

                if del_btn:
                    st.session_state.bons = st.session_state.bons.drop(index=idx).reset_index(drop=True)
                    save_bons_df(); st.success("🗑️ Bon supprimé"); st.rerun()

ZONE_HEX = {  # couleurs proches de tes ZONE_RGB
    "75":"#eeeeee","91":"#3b82f6","92":"#ef4444","93":"#8b5cf6","94":"#784830",
    "95":"#10b981","77":"#fb923c","78":"#f59e0b","60":"#7b3fb2"
}

TOUR_HEX = {
    "1": "#5b8cff",  # bleu
    "2": "#10b981",  # vert
    "3": "#ef4444",  # rouge
    "4": "#f59e0b",  # orange
    "5": "#8b5cf6",  # violet
}


import zlib  # tout en haut du fichier si tu préfères

def _row_uid(row):
    # Si un uid numérique existe déjà, on le garde
    if pd.notna(row.get("uid")):
        try:
            return int(row["uid"])
        except Exception:
            pass
    # Sinon, UID déterministe basé sur id_bon (ou l'index)
    key = clean_str(row.get("id_bon", "")) or f"row-{int(row.name)}"
    return zlib.crc32(key.encode("utf-8")) & 0xffffffff


def _route_toggle_in_active(uid: int):
    """Ajoute/retire un point du tracé pour la tournée active."""
    t = st.session_state.active_route_tour
    cur = st.session_state.route_sels.get(t, [])
    if uid in cur:
        st.session_state.route_sels[t] = [u for u in cur if u != uid]
    else:
        st.session_state.route_sels[t] = cur + [uid]

def _route_coords_for_polyline(df: pd.DataFrame, uids):
    pts=[]
    for u in uids:
        r = df[df["uid"]==u]
        if not r.empty and pd.notna(r.iloc[0]["lat"]) and pd.notna(r.iloc[0]["lon"]):
            pts.append((float(r.iloc[0]["lat"]), float(r.iloc[0]["lon"])))
    return pts

# -----------------------------------------------------------------------------
#          AFFECTATION — carte + DnD + bouton Patron (traits + listes sous carte)
# -----------------------------------------------------------------------------
if tab_is("🗂️ Affectation"):
    st.markdown("# Affectation")

    if "bons" not in st.session_state or st.session_state.bons.empty:
        st.info("Aucun bon pour l’instant. Ajoute des bons dans l’onglet **Saisie**.")
        st.stop()


    def norm_tournee(v: object) -> str:
        s = clean_str(v)
        if not s: return ""
        try: return str(int(float(s)))
        except Exception: return s

    def uniq_label_for_dnd(row):
        base = bon_chip(row)
        uid = str(int(row.get("uid")) if pd.notna(row.get("uid")) else clean_str(row.get("id_bon")))
        return f"{base}\u2063{uid}"

    def strip_suffix(lbl: str) -> str:
        return lbl.split("\u2063")[0]

    def primary_zone_any(r):
        d = clean_str(r.get("dept_affichage"))
        if d: return d
        raw = clean_str(r.get("zones"))
        if raw:
            z0 = raw.split(",")[0].strip()
            if z0: return z0
        return clean_str(r.get("zone_bucket"))

    def render_board_readonly(zone_containers, tour_containers):
        st.markdown("### Zones & Tournées (lecture seule)")
        zcols = st.columns(max(1, min(4, len(zone_containers) or 1)))
        for i, cont in enumerate(zone_containers):
            with zcols[i % len(zcols)]:
                st.markdown(f"**{cont['header']}**")
                if cont["items"]:
                    for it in cont["items"]:
                        st.markdown(f"<span class='combo'>{strip_suffix(it)}</span>", unsafe_allow_html=True)
                else:
                    st.caption("—")
        st.markdown("---")
        tcols = st.columns(max(1, min(4, len(tour_containers) or 1)))
        for i, cont in enumerate(tour_containers):
            with tcols[i % len(tcols)]:
                st.markdown(f"**{cont['header']}**")
                if cont["items"]:
                    for it in cont["items"]:
                        st.markdown(f"<span class='combo'>{strip_suffix(it)}</span>", unsafe_allow_html=True)
                else:
                    st.caption("—")
    
    # === CARTE UNIQUE (Folium) : cliquable + trajets Patron + points ===
    st.markdown("### 🗺️ Carte des bons ")

    if not HAVE_FOLIUM:
        st.error("Installe :  py -m pip install -U streamlit-folium folium")
    else:
        df_all = st.session_state.bons.copy()
        df_all["zone_1"] = df_all.apply(primary_zone_any, axis=1)
        df_pts = df_all.dropna(subset=["lat","lon"]).copy()

        if df_pts.empty:
            st.info("Aucun point géolocalisé à afficher.")
        else:
            # Assure un uid entier pour TOUS les points (même si NaN dans le CSV)
            df_pts["uid"] = pd.to_numeric(df_pts["uid"], errors="coerce")
            mask_na_uid = df_pts["uid"].isna()
            if mask_na_uid.any():
                df_pts.loc[mask_na_uid, "uid"] = df_pts.loc[mask_na_uid].apply(_row_uid, axis=1)
            df_pts["uid"] = df_pts["uid"].astype(int)

            # centre carte
            center = [float(df_pts["lat"].mean()), float(df_pts["lon"].mean())]

            # Fond sombre façon “Dark Matter”
            tiles = "CartoDB dark_matter"
            fmap = folium.Map(location=center, zoom_start=10, control_scale=True, tiles=tiles)

            # --- 1) Trajets proposés (Patron) en PathLayer Folium ---
            plan_paths = st.session_state.get("patron_paths", [])
            for P in plan_paths:
                path = P.get("path") or []
                # path = [[lon,lat], ...] → Folium veut (lat,lon)
                latlon = [(lat, lon) for lon, lat in path]
                color = "#{:02x}{:02x}{:02x}".format(*P.get("color", [180,180,180]))
                if len(latlon) >= 2:
                    folium.PolyLine(latlon, weight=5, opacity=0.9, color=color).add_to(fmap)

            # --- 2) Points cliquables (tous les bons) ---
            for _, r in df_pts.iterrows():
                uid = _row_uid(r)
                zone = primary_zone_any(r)
                color = ZONE_HEX.get(zone, "#b4b4b4")

                # uid dans au moins un des tracés actifs ?
                in_any_route = any(uid in st.session_state.route_sels.get(t, []) for t in TOURNEES)
                selected = in_any_route
                edge = "#ffffff" if selected else color


                folium.CircleMarker(
                    location=(float(r["lat"]), float(r["lon"])),
                    radius=9 if selected else 7,
                    color=edge, weight=3 if selected else 2,
                    fill=True, fill_color=color, fill_opacity=0.95,
                    # survol = infos
                    tooltip=f"{clean_str(r['client'])} • {clean_str(r['ville'])} • {zone}",
                    # clic = UID (on le lira via st_folium)
                    popup=str(uid),
                ).add_to(fmap)


            # --- 3) Tracé en cours (sélection par clic) ---
            # Tracés en cours : un par tournée
            for t in TOURNEES:
                uids = st.session_state.route_sels.get(t, [])
                route_pts = _route_coords_for_polyline(df_pts, uids)
                if len(route_pts) >= 2:
                    folium.PolyLine(route_pts, weight=5, opacity=0.95, color=TOUR_HEX.get(t, "#5b8cff")).add_to(fmap)


            mn = st.session_state.get("map_nonce", 0)
            out = st_folium(
                fmap,
                height=560,
                width=None,
                returned_objects=["last_object_clicked", "last_object_clicked_popup"],
                key=f"fmap_{mn}",
            )


            # Récupère l'UID cliqué (depuis le popup du point)
            uid_clicked = None

            # 1) Preferred: popup payload
            popup_val = out.get("last_object_clicked_popup")
            if isinstance(popup_val, (bytes, bytearray)):
                popup_val = popup_val.decode("utf-8", errors="ignore")
            if isinstance(popup_val, (str, int)) and str(popup_val).strip().isdigit():
                uid_clicked = int(str(popup_val).strip())
            else:
                # Fallback for older plugin versions: sometimes the popup is nested
                loc = out.get("last_object_clicked")
                if isinstance(loc, dict) and str(loc.get("popup", "")).strip().isdigit():
                    uid_clicked = int(str(loc["popup"]).strip())

            if uid_clicked is not None:
                _route_toggle_in_active(uid_clicked)
                st.rerun()



            # ---- Barre d’actions liée aux tracés ----
            st.markdown("#### ✨ Tracés en cours (multi-tournées)")

            c0, cA, cB, cC, cD = st.columns([2,1,1,2,2])

            # Choix de la tournée qu’on MODIFIE via la carte
            st.session_state.active_route_tour = c0.selectbox(
                "Tournée en édition (les clics s'ajoutent ici)",
                TOURNEES,
                index=TOURNEES.index(st.session_state.active_route_tour) if st.session_state.active_route_tour in TOURNEES else 0,
                key="active_route_sel"
            )

            # Boutons pour la tournée active uniquement
            if cA.button("↩️ Retirer le dernier (tournée active)"):
                t = st.session_state.active_route_tour
                cur = st.session_state.route_sels.get(t, [])
                if cur:
                    st.session_state.route_sels[t] = cur[:-1]
                    st.rerun()

            if cB.button("🧹 Vider le tracé (tournée active)"):
                t = st.session_state.active_route_tour
                st.session_state.route_sels[t] = []
                st.rerun()

            # Appliquer un tracé → sur sa tournée (même nom)
            tgt_tour = st.session_state.active_route_tour
            btn_apply_trace = cD.button("✅ Appliquer le tracé sur sa tournée")

            # Listing rapide
            chips_all = []
            for t in TOURNEES:
                uids = st.session_state.route_sels.get(t, [])
                if uids:
                    names = []
                    for u in uids:
                        rmatch = df_pts[df_pts["uid"] == u]
                        if not rmatch.empty:
                            r = rmatch.iloc[0]
                            names.append(f"{clean_str(r['client'])} ({clean_str(r['ville'])})")
                    if names:
                        st.write(f"**Tournée {t}** — " + " → ".join(names))

            if btn_apply_trace:
                uids = st.session_state.route_sels.get(tgt_tour, [])
                if not uids:
                    st.warning("Le tracé de la tournée active est vide.")
                else:
                    ordre = 1
                    # index rapide uid → index dataframe
                    by_uid = {int(r["uid"]): i for i, r in st.session_state.bons.reset_index().iterrows()}
                    moved = 0
                    for u in uids:
                        i = by_uid.get(int(u))
                        if i is None: 
                            continue
                        idx = st.session_state.bons.index[i]
                        st.session_state.bons.at[idx, "tournee"] = str(tgt_tour)
                        st.session_state.bons.at[idx, "ordre"] = ordre
                        ordre += 1
                        moved += 1
                    save_bons_df()
                    st.success(f"✅ {moved} étiquette(s) → Tournée {tgt_tour}")

                    # on garde le tracé (utile pour itérer), sinon décommente la remise à zéro :
                    # st.session_state.route_sels[tgt_tour] = []
                    st.session_state.pop("board_zones_tournees_v9", None)
                    st.session_state["map_nonce"] = st.session_state.get("map_nonce", 0) + 1
                    st.rerun()




    def _uids_from_index_order(df: pd.DataFrame, idxs: list[int]) -> list[int]:
        """Transforme une liste d'index de df en liste d'UID pour le tracé."""
        out = []
        for i in idxs:
            try:
                u = int(df.at[i, "uid"])
                out.append(u)
            except Exception:
                pass
        return out


    # bouton de proposition de tournées (Patron) — NE TOUCHE PAS AU GÉOCODAGE
    c_btn, _ = st.columns([1,3])
    with c_btn:
        if st.button("🤖 Proposer des tournées (Patron)"):
            try:
                _ensure_depot()
                plan = patron_build_plan(st.session_state.bons)
                paths = patron_plan_paths(st.session_state.bons, plan)
                annot = patron_plan_annotations(st.session_state.bons, plan)
                st.session_state["patron_plan"] = plan
                st.session_state["patron_paths"] = paths
                st.session_state["patron_annot"] = annot

                # 🔵 Charger automatiquement les tracés des tournées proposées (toutes)
                st.session_state.route_sels = {t: [] for t in TOURNEES}
                tours = plan.get("tours", []) or []
                for T in tours:
                    tlab = str(T.get("t",""))
                    idxs = T.get("order", []) or []
                    st.session_state.route_sels[tlab] = _uids_from_index_order(st.session_state.bons, idxs)

                # Option : se placer directement sur la première tournée en édition
                if tours:
                    st.session_state.active_route_tour = str(tours[0].get("t","1"))


                st.success("Proposition calculée. Les trajets sont dessinés sur la carte et listés sous la carte.")
            except Exception as e:
                st.error(f"Proposition impossible: {e}")

    # === LISTES DE PROPOSITIONS SOUS LA CARTE ===================================
    st.markdown("#### 📋 Propositions de tournées (Patron) — sous la carte")
    plan = st.session_state.get("patron_plan")
    annot = st.session_state.get("patron_annot")
    if plan and isinstance(plan, dict) and annot is not None:
        # Afficher chaque tournée proposée avec ses bons et un bouton "Appliquer"
        for T in plan.get("tours", []):
            idxs = T.get("order", []) or []
            if not idxs:
                continue
            sub = st.session_state.bons.loc[idxs].copy()
            sub = sub.assign(_ord=range(1, len(sub)+1))
            tot_big = int(sub["n_big"].fillna(0).sum())
            tot_small = int(sub["n_small"].fillna(0).sum())
            tlabel = str(T.get("t", ""))
            zone = T.get("zone") or "?"
            truck_id = T.get("truck", "")
            est_km = round(_route_len_with_return(st.session_state.bons, idxs), 1)
            tot_weight = int(sub["poids_kg"].fillna(0).sum())

            with st.expander(
                f"🧭 Tournée {tlabel} • Camion {truck_id} (zone {zone}) — "
                f"{len(sub)} bon(s) • {tot_big} BIG / {tot_small} SMALL • ~{est_km} km • {tot_weight} kg",
                expanded=False
            ):

                # liste des bons
                for _, r in sub.iterrows():
                    chip = bon_chip(r)
                    st.markdown(f"<span class='combo'>{chip}</span>", unsafe_allow_html=True)

                c1, c2 = st.columns([1,1])
                with c1:
                    apply_one = st.button(f"✅ Appliquer cette tournée {tlabel}", key=f"apply_prop_{tlabel}")
                with c2:
                    clear_one = st.button(f"↩️ Réinitialiser proposition {tlabel}", key=f"clear_prop_{tlabel}")

                if apply_one:
                    # On applique uniquement aux bons qui ne sont pas déjà envoyés à un camion
                    mask_apply = st.session_state.bons.index.isin(idxs) & (st.session_state.bons["camion_id"].apply(clean_str) == "")
                    affected = st.session_state.bons[mask_apply].index.tolist()
                    if affected:
                        # Ordre = proposition
                        ordre_map = {idx: i+1 for i, idx in enumerate(idxs)}
                        for idx in affected:
                            st.session_state.bons.at[idx, "tournee"] = tlabel
                            st.session_state.bons.at[idx, "ordre"] = ordre_map.get(idx)
                        save_bons_df()
                        st.success(f"✅ Proposition appliquée pour {len(affected)} bon(s) → Tournée {tlabel}")
                        st.rerun()
                    else:
                        st.info("Rien à appliquer (déjà affecté à un camion ou aucun bon concerné).")

                if clear_one:
                    # Ne touche pas aux données, juste enlève l'affichage des propositions pour cette tournée
                    # (on retire la tournée T du plan en session)
                    new_tours = [TT for TT in plan.get("tours", []) if str(TT.get("t","")) != tlabel]
                    st.session_state["patron_plan"] = {"tours": new_tours}
                    # Recalcule les chemins/annotations
                    st.session_state["patron_paths"] = patron_plan_paths(st.session_state.bons, st.session_state["patron_plan"])
                    st.session_state["patron_annot"] = patron_plan_annotations(st.session_state.bons, st.session_state["patron_plan"])
                    st.rerun()

    # === Saisie & géocodage MANUELS des adresses manquantes ======================
    c_geo, _ = st.columns([1, 3])
    if "show_geo_manual" not in st.session_state:
        st.session_state.show_geo_manual = False
    with c_geo:
        if st.button("✏️ Saisir les adresses manquantes", use_container_width=True):
            st.session_state.show_geo_manual = not st.session_state.show_geo_manual

    df_missing = st.session_state.bons[
        st.session_state.bons["lat"].isna() | st.session_state.bons["lon"].isna()
    ].copy()

    if st.session_state.show_geo_manual:
        st.markdown("#### 📍 Adresses à compléter")
        if df_missing.empty:
            st.success("Tout est déjà géocodé ✅")
        else:
            with st.form("manual_missing_geocode"):
                rows_ui = []
                for _, r in df_missing.reset_index().iterrows():
                    idx = int(r["index"])
                    id_bon = str(r["id_bon"])
                    client = clean_str(r.get("client"))
                    ville  = clean_str(r.get("ville"))
                    adr    = clean_str(r.get("adresse"))
                    default_addr = _compose_address_for_geocode(adr, ville)

                    st.write("---")
                    st.write(f"**{id_bon}** — {client or 'Client ?'}")

                    colA, colB, colC = st.columns([2, 1, 1])
                    with colA:
                        addr_input = st.text_input(
                            "Adresse pour géocodage",
                            key=f"geo_addr_{idx}",
                            value=default_addr,
                            placeholder="ex: 11 Avenue Lénine 93120 La Courneuve France"
                        )
                    with colB:
                        lat_input = st.text_input("Latitude (optionnel)", key=f"geo_lat_{idx}", value="")
                    with colC:
                        lon_input = st.text_input("Longitude (optionnel)", key=f"geo_lon_{idx}", value="")
                    rows_ui.append((idx, addr_input, lat_input, lon_input))

                c_ok, c_reset = st.columns([1, 1])
                submit = c_ok.form_submit_button("📌 Géocoder ces adresses", use_container_width=True)
                reset_flag = c_reset.form_submit_button("↺ Vider les champs (lat/lon)", use_container_width=True)

            if reset_flag:
                for idx, _, _, _ in rows_ui:
                    st.session_state[f"geo_lat_{idx}"] = ""
                    st.session_state[f"geo_lon_{idx}"] = ""
                st.rerun()

            if submit:
                if not HAVE_GEOPY:
                    st.warning("Installe `geopy` :  py -m pip install geopy")
                else:
                    ok_cnt, manual_cnt, err_cnt = 0, 0, 0
                    for idx, addr_input, lat_input, lon_input in rows_ui:
                        lat_input = clean_str(lat_input)
                        lon_input = clean_str(lon_input)

                        if lat_input and lon_input:
                            try:
                                st.session_state.bons.at[idx, "lat"] = float(lat_input)
                                st.session_state.bons.at[idx, "lon"] = float(lon_input)
                                manual_cnt += 1
                                continue
                            except Exception:
                                err_cnt += 1
                                continue

                        addr_try = clean_str(addr_input)
                        if addr_try:
                            try:
                                loc = geocode_address(addr_try)
                                if loc:
                                    st.session_state.bons.at[idx, "lat"] = float(loc.latitude)
                                    st.session_state.bons.at[idx, "lon"] = float(loc.longitude)
                                    ok_cnt += 1
                                else:
                                    err_cnt += 1
                            except Exception:
                                err_cnt += 1

                    save_bons_df()
                    st.success(f"✅ {ok_cnt} géocodé(s) • ✍️ {manual_cnt} saisi(s) manuellement • ⚠️ {err_cnt} échec(s)")
                    st.rerun()
    else:
        if df_missing.empty:
            st.caption("Aucune adresse manquante.")
        else:
            st.caption(f"{len(df_missing)} bon(s) restant(s) à géocoder.")

    # === Board Zones & Tournées (bons sans camion)
    st.session_state.bons["tournee"] = st.session_state.bons["tournee"].apply(norm_tournee)
    df_all = st.session_state.bons
    df_board = df_all[df_all["camion_id"].apply(clean_str) == ""].copy()
    df_board["zone_1"] = df_board.apply(primary_zone_any, axis=1)

    containers, containers_zones, containers_tours = [], [], []
    display_to_id = {}

    not_in_tour = df_board[df_board["tournee"].apply(clean_str) == ""].copy()
    zones_connues = [z for z in ZONES if z in set(df_board["zone_1"].unique())]
    for z in zones_connues:
        header = f"Zone {z} {ZONE_EMOJI.get(z, '⬜️')}"
        sub = not_in_tour[not_in_tour["zone_1"] == z].copy()
        items = []
        for _, r in sub.iterrows():
            label_uniq = uniq_label_for_dnd(r); display_to_id[label_uniq] = str(r["id_bon"])
            items.append(label_uniq)
        cont = {"header": header, "items": items}
        containers.append(cont); containers_zones.append(cont)

    def tour_totals_unassigned(t: str):
        sub = df_board[df_board["tournee"] == t]
        return int(sub["n_big"].fillna(0).sum()), int(sub["n_small"].fillna(0).sum())

    for t in TOURNEES:
        sub = df_board[df_board["tournee"] == t].copy()
        items = []
        for _, r in sub.iterrows():
            label_uniq = uniq_label_for_dnd(r); display_to_id[label_uniq] = str(r["id_bon"])
            items.append(label_uniq)
        tb, ts = tour_totals_unassigned(t)
        head = f"🧱 Tournée {t} — {tb} BIG / {ts} SMALL"
        cont = {"header": head, "items": items}
        containers.append(cont); containers_tours.append(cont)

    use_dnd = HAVE_DND and not st.session_state.get("DND_BROKEN", False)

    if use_dnd:
        prev_loc = {}
        for z in zones_connues:
            header = f"Zone {z} {ZONE_EMOJI.get(z, '⬜️')}"
            sub = df_board[(df_board["tournee"].apply(clean_str) == "") & (df_board["zone_1"] == z)]
            for _, r in sub.iterrows(): prev_loc[str(r["id_bon"])] = header
        for t in TOURNEES:
            header = f"Tournée {t}"; sub = df_board[df_board["tournee"] == t]
            for _, r in sub.iterrows(): prev_loc[str(r["id_bon"])] = header

        new = sort_items_safe(items=containers, multi_containers=True, key="board_zones_tournees_v9")

        if st.session_state.get("DND_BROKEN", False) or not isinstance(new, list) or not new:
            st.warning("Drag & drop indisponible — affichage lecture seule.")
            render_board_readonly(containers_zones, containers_tours)
        else:
            id_to_tour, id_to_order, id_to_header = {}, {}, {}
            for cont in new:
                header = cont.get("header", ""); target_t = ""
                if header.startswith("🧱 Tournée "):
                    try: target_t = header.split()[2]
                    except Exception: target_t = ""
                for idx_pos, lbl in enumerate(cont.get("items", []), start=1):
                    bid = display_to_id.get(lbl)
                    if not bid: continue
                    id_to_tour[bid] = target_t
                    id_to_order[bid] = idx_pos if target_t else None
                    id_to_header[bid] = header

            any_move = any(id_to_header.get(bid, old) != old for bid, old in prev_loc.items())
            if any_move:
                changed = False
                for i, r in st.session_state.bons.iterrows():
                    bid = str(r["id_bon"])
                    if bid not in prev_loc: continue
                    new_t = norm_tournee(id_to_tour.get(bid, ""))
                    old_t = norm_tournee(r.get("tournee"))
                    if new_t != old_t:
                        st.session_state.bons.at[i, "tournee"] = new_t
                        changed = True
                    st.session_state.bons.at[i, "ordre"] = id_to_order.get(bid)
                if changed:
                    save_bons_df(); st.success("✅ Affectations mises à jour"); st.rerun()
    else:
        st.warning("Drag & drop indisponible — lecture seule + formulaire simple.")
        render_board_readonly(containers_zones, containers_tours)

        with st.form("fallback_assign_compact"):
            c1, c2 = st.columns([2, 1])
            with c1:
                not_in_tour = df_board[df_board["tournee"].apply(clean_str) == ""]
                options = { f"{strip_suffix(uniq_label_for_dnd(r))}": str(r["id_bon"])
                           for _, r in not_in_tour.iterrows() }
                sel = st.multiselect("Sélectionne des bons à affecter", list(options.keys()))
            with c2:
                tgt = st.selectbox("Vers tournée", TOURNEES)
            ok_fb = st.form_submit_button("Affecter")

        if ok_fb and sel:
            ids = {options[s] for s in sel}
            cur = df_board[df_board["tournee"] == tgt]
            start = int(cur["ordre"].fillna(0).max() or 0) + 1
            for i in st.session_state.bons.index:
                if str(st.session_state.bons.at[i, "id_bon"]) in ids:
                    st.session_state.bons.at[i, "tournee"] = tgt
                    st.session_state.bons.at[i, "ordre"] = start
                    start += 1
            save_bons_df(); st.success("✅ Bons affectés"); st.rerun()

    # --- Boutons : renvoyer une tournée vers son bloc Zone ------------------------
    st.markdown("#### ↩️ Renvoyer les étiquettes d’une tournée vers leur zone")

    cols = st.columns(min(5, len(TOURNEES)))  # petite rangée de boutons
    ci = 0
    for t in TOURNEES:
        # on ne s'occupe que des bons NON envoyés à un camion (camion_id vide)
        mask = (st.session_state.bons["tournee"] == t) & (st.session_state.bons["camion_id"].apply(clean_str) == "")
        n = int(mask.sum())
        if n == 0:
            continue  # pas de bouton si tournée vide

        col = cols[ci % len(cols)]
        ci += 1

        if col.button(f"↩️ Tournée {t} → Zones ({n})", key=f"sendback_tour_{t}"):
            # on "renvoie" : la Zone est déjà portée par dept_affichage/zone_bucket, 
            # le board les remettra automatiquement au bon endroit
            st.session_state.bons.loc[mask, ["tournee", "ordre"]] = ["", None]
            save_bons_df()
            # on force le rafraîchissement du board DnD et de la carte au besoin
            st.session_state.pop("board_zones_tournees_v9", None)
            st.session_state["map_nonce"] = st.session_state.get("map_nonce", 0) + 1
            st.success(f"↩️ {n} étiquette(s) renvoyée(s) vers leur zone (Tournée {t}).")
            st.rerun()


    st.markdown("---")

    # === TOURNÉES -> CAMIONS
    st.markdown("### Tournée → Camion (validation combos)")
    trucks_df = st.session_state.trucks

    def reset_tour(t: str):
        st.session_state.tournee_truck.pop(t, None)
        idx_bons = st.session_state.bons[st.session_state.bons["tournee"] == t].index
        if not idx_bons.empty:
            st.session_state.bons = st.session_state.bons.drop(index=idx_bons).reset_index(drop=True)
        for k in list(st.session_state.keys()):
            if isinstance(k, str) and (k.startswith(f"truck_{t}") or k.startswith(f"sent_{t}") or k.startswith(f"apply_{t}")):
                st.session_state.pop(k, None)
        for k in list(st.session_state.keys()):
            if isinstance(k, str) and (k.startswith("board_") or k.startswith("DND") or k.startswith("board_zones_tournees") or k.startswith("geo_")):
                st.session_state.pop(k, None)
        save_bons_df()
        st.info(f"🔁 Tournée {t} réinitialisée — les bons de la tournée ont été supprimés.")
        st.rerun()

    def finalize_tour(t: str) -> int:
        """
        Archive la tournée t dans history.csv puis supprime tout ce qui s’y rattache
        dans l’app courante (étiquettes, ordre, camion, tracés, traits Patron).
        Retourne le nombre d’étiquettes archivées.
        """
        t = clean_str(t)
        if not t:
            return 0

        df = st.session_state.bons
        mask = df["tournee"].astype(str) == t
        if not bool(mask.any()):
            return 0

        # qui est le camion ?
        truck = st.session_state.tournee_truck.get(t, "")

        # 1) ARCHIVE
        now = pd.Timestamp.utcnow().isoformat()
        to_archive = df[mask].copy()
        to_archive["finished_at"] = now
        to_archive["tournee_label"] = t
        to_archive["truck_id"] = clean_str(truck)

        hist = load_history_df()
        hist = pd.concat([hist, to_archive[_history_columns()]], ignore_index=True)
        save_history_df(hist)

        # 2) NETTOIE L’APP COURANTE
        #    - enlève ces lignes des bons actifs
        st.session_state.bons = df[~mask].reset_index(drop=True)
        save_bons_df()

        #    - oublie l’affectation camion
        st.session_state.tournee_truck.pop(t, None)

        #    - vide le tracé manuel de cette tournée
        if "route_sels" in st.session_state:
            st.session_state.route_sels[t] = []

        #    - retire la tournée des propositions Patron + traits
        if "patron_plan" in st.session_state and isinstance(st.session_state.patron_plan, dict):
            tours = st.session_state.patron_plan.get("tours", []) or []
            tours = [TT for TT in tours if str(TT.get("t","")) != t]
            st.session_state["patron_plan"] = {"tours": tours}
            # recalc chemins/annotations à blanc pour cette tournée
            st.session_state["patron_paths"] = patron_plan_paths(st.session_state.bons, st.session_state["patron_plan"])
            st.session_state["patron_annot"] = patron_plan_annotations(st.session_state.bons, st.session_state["patron_plan"])

        #    - force le refresh de la carte et du board DnD
        st.session_state["map_nonce"] = st.session_state.get("map_nonce", 0) + 1
        st.session_state.pop("board_zones_tournees_v9", None)

        return int(mask.sum())


    for t in TOURNEES:
        sub = st.session_state.bons[st.session_state.bons["tournee"] == t]
        if sub.empty: continue
        tot_big = int(sub["n_big"].fillna(0).sum())
        tot_small = int(sub["n_small"].fillna(0).sum())
        with st.expander(f"Tournée {t} — {tot_big} BIG / {tot_small} SMALL", expanded=True):
            tid_cur = st.session_state.tournee_truck.get(t)
            choices = [""] + trucks_df["id"].tolist()
            idx = choices.index(tid_cur) if tid_cur in choices else 0
            tid = st.selectbox("Camion", choices, index=idx, key=f"truck_{t}")

            if st.button("Affecter/Mettre à jour", key=f"apply_{t}"):
                if tid == "":
                    st.session_state.tournee_truck[t] = None
                    idx_bons = st.session_state.bons[st.session_state.bons["tournee"] == t].index
                    st.session_state.bons.loc[idx_bons, "camion_id"] = ""
                    save_bons_df(); st.success("Camion retiré"); st.rerun()
                else:
                    tr = trucks_df.set_index("id").loc[tid].to_dict()
                    if fits_any_combo(tr, tot_big, tot_small):
                        st.session_state.tournee_truck[t] = tid
                        idx_bons = st.session_state.bons[st.session_state.bons["tournee"] == t].index
                        st.session_state.bons.loc[idx_bons, "camion_id"] = tid
                        save_bons_df(); st.success("✅ Tournée affectée au camion"); st.rerun()
                    else:
                        st.error(f"Impossible : {tot_big} BIG / {tot_small} SMALL dépasse les combos du camion **{tid}**.")

    st.markdown("---")
    st.markdown("## Tournées affectées (récap)")

    any_assigned = False
    for t in TOURNEES:
        tid_cur = st.session_state.tournee_truck.get(t)
        if not tid_cur: continue

        sub_all = st.session_state.bons[st.session_state.bons["tournee"] == t].copy()
        if sub_all.empty: continue

        any_assigned = True
        sub_all = sub_all.sort_values(by=["ordre", "is_restaurant"], ascending=[True, False])
        tot_big = int(sub_all["n_big"].fillna(0).sum())
        tot_small = int(sub_all["n_small"].fillna(0).sum())
        st.markdown(f"**Tournée {t} → Camion {tid_cur}** — {len(sub_all)} bon(s) • {tot_big} BIG / {tot_small} SMALL")

        cols_show = ["ordre", "id_bon", "client", "ville", "adresse", "poids_kg", "n_big", "n_small", "is_restaurant"]
        st.dataframe(sub_all[cols_show], use_container_width=True, hide_index=True,
                     height=min(340, 40 + 28 * len(sub_all)))
        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    if not any_assigned:
        st.info("Aucune tournée n'est encore affectée à un camion.")


# -----------------------------------------------------------------------------
#                                   EXPORTS
# -----------------------------------------------------------------------------
if tab_is("📦 Exports"):
    st.markdown("# 📦 Exports")
    df = st.session_state.bons.copy()

    # Tournées affectées (camion_id non vide)
    assigned = df[df["camion_id"].apply(clean_str) != ""].copy()
    if assigned.empty:
        st.info("Aucune tournée affectée à un camion pour le moment.")
        st.stop()

    # petit helper PDF
    def make_feuille_route_pdf(tlabel: str, truck_id: str, sdf: pd.DataFrame) -> bytes:
        if not HAVE_REPORTLAB:
            return b""
        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        W, H = A4

        # ---------- helpers ----------
        def fit_txt(txt: str, max_w_mm: float, font="Helvetica", size=9):
            """Raccourcit avec … pour tenir dans max_w_mm."""
            max_w = max_w_mm * mm
            s = clean_str(txt)
            if not s: return ""
            w = c.stringWidth(s, font, size)
            if w <= max_w:
                return s
            # garder un peu de marge pour "…"
            ell = "…"
            ell_w = c.stringWidth(ell, font, size)
            # recherche décroissante
            lo, hi = 0, len(s)
            res = ""
            while lo < hi:
                mid = (lo + hi) // 2
                cand = s[:mid].rstrip() + ell
                if c.stringWidth(cand, font, size) <= max_w - 0.5*mm:
                    res = cand
                    lo = mid + 1
                else:
                    hi = mid
            return res or ell

        def draw_right(x_right_mm: float, y, txt: str, font="Helvetica", size=9):
            c.setFont(font, size)
            xr = x_right_mm * mm
            w = c.stringWidth(txt, font, size)
            c.drawString(xr - w, y, txt)

        # ---------- entête ----------
        c.setFont("Helvetica-Bold", 22)
        c.drawString(18*mm, H-22*mm, f"Feuille de route — Tournée {tlabel} — Camion {truck_id}")
        tot_big = int(sdf["n_big"].fillna(0).sum())
        tot_small = int(sdf["n_small"].fillna(0).sum())
        c.setFont("Helvetica", 12)
        c.drawString(18*mm, H-30*mm, f"{len(sdf)} bon(s) — {tot_big} BIG / {tot_small} SMALL")

        # ---------- tableau ----------
        y = H - 42*mm
        row_h = 6.2*mm
        head_y = y

        # largeurs (mm) — ajustées pour éviter tout recouvrement
        col_w = {
            "Ordre": 12, "Bon": 24, "Client": 46, "Ville": 40, "Adresse": 84,
            "Poids": 18, "BIG": 12, "SMALL": 16
        }
        cols = ["Ordre","Bon","Client","Ville","Adresse","Poids","BIG","SMALL"]

        # positions x (bord gauche)
        x = 10*mm
        x_pos = {cols[0]: x}
        for i in range(1, len(cols)):
            prev = cols[i-1]
            x_pos[cols[i]] = x_pos[prev] + col_w[prev]*mm

        # entête
        c.setFont("Helvetica-Bold", 10)
        for h in cols:
            if h in ("Poids","BIG","SMALL"):
                # alignée à droite
                draw_right((x_pos[h]/mm) + col_w[h], y, h, "Helvetica-Bold", 10)
            else:
                c.drawString(x_pos[h], y, h)
        y -= row_h

        c.setFont("Helvetica", 9)

        for _, r in sdf.iterrows():
            if y < 18*mm:  # nouvelle page
                c.showPage()
                c.setFont("Helvetica-Bold", 10)
                y = H - 20*mm
                for h in cols:
                    if h in ("Poids","BIG","SMALL"):
                        draw_right((x_pos[h]/mm) + col_w[h], y, h, "Helvetica-Bold", 10)
                    else:
                        c.drawString(x_pos[h], y, h)
                y -= row_h
                c.setFont("Helvetica", 9)

            row = {
                "Ordre": str(int(r.get("ordre") or 0)),
                "Bon":   fit_txt(r.get("id_bon"),   col_w["Bon"]),
                "Client":fit_txt(r.get("client"),  col_w["Client"]),
                "Ville": fit_txt(r.get("ville"),   col_w["Ville"]),
                "Adresse":fit_txt(r.get("adresse"),col_w["Adresse"]),
                "Poids": str(int(float(r.get("poids_kg") or 0))),
                "BIG":   str(int(r.get("n_big") or 0)),
                "SMALL": str(int(r.get("n_small") or 0)),
            }

            # colonnes texte (gauche)
            for k in ("Ordre","Bon","Client","Ville","Adresse"):
                c.drawString(x_pos[k], y, row[k])

            # colonnes numériques (droite)
            for k in ("Poids","BIG","SMALL"):
                draw_right((x_pos[k]/mm) + col_w[k], y, row[k])

            y -= row_h

        c.showPage()
        c.save()
        return buf.getvalue()


    # groupement par tournée + camion
    groups = assigned.groupby(["tournee", "camion_id"], dropna=False)
    for (tlabel, truck_id), sdf in groups:
        sdf = sdf.sort_values(by=["ordre", "is_restaurant"], ascending=[True, False]).copy()

        tot_big = int(sdf["n_big"].fillna(0).sum())
        tot_small = int(sdf["n_small"].fillna(0).sum())
        st.markdown(f"## Tournée {tlabel} → Camion {truck_id} — {len(sdf)} bon(s) • {tot_big} BIG / {tot_small} SMALL")

        # Aperçu rapide
        cols_show = ["ordre","id_bon","client","ville","adresse","poids_kg","n_big","n_small","is_restaurant"]
        st.dataframe(sdf[cols_show], use_container_width=True, hide_index=True, height=min(320, 40+28*len(sdf)))

        # Boutons d’export
        c1, c2 = st.columns([1,1])
        # CSV
        csv_bytes = to_csv_str(sdf).encode("utf-8")
        c1.download_button(
            "⬇️ Export CSV (tournée)",
            data=csv_bytes,
            file_name=f"tournee_{tlabel}_{truck_id}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"csv_{tlabel}_{truck_id}"
        )
        c3 = st.container()
        if c3.button(f"✅ Finir et archiver la tournée {tlabel}", key=f"done_{tlabel}_{truck_id}"):
            n = finalize_tour(str(tlabel))
            st.success(f"Tournée {tlabel} archivée ({n} bon(s))")
            st.rerun()

        # PDF
        if HAVE_REPORTLAB:
            pdf_bytes = make_feuille_route_pdf(str(tlabel), str(truck_id), sdf)
            c2.download_button(
                "🧾 Feuille de route (PDF)",
                data=pdf_bytes,
                file_name=f"feuille_route_{tlabel}_{truck_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"pdf_{tlabel}_{truck_id}"
            )
        else:
            c2.info("Installe `reportlab` pour générer le PDF :  `py -m pip install reportlab`")

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


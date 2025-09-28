# normalize_dataset.py
import cv2, os, glob, json, numpy as np
from tqdm import tqdm

SRC = "dataset"                # raiz com imagens originais (BGR ou já cinza)
DST = "dataset-norm-2"         # saída
OUT_W, OUT_H = 512, 384

# --------- NORMALIZAÇÃO ---------
MODE = "dataset_fixed"         # "per_frame" | "dataset_fixed"
P_LO, P_HI = 5, 95             # percentis de recorte
CLAHE = False                  # equalização local após normalizar
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

# --------- MÁSCARA / RECORTE (pra não contaminar percentis) ---------
EXCLUDE_TOP_RATIO = 0.00       # ex.: 0.25 se tiver teto muito quente
EXCLUDE_BOTTOM_PX = 22         # remove logo/rodapé (ajuste conforme dataset)
EXCLUDE_LEFT_PX = 0
EXCLUDE_RIGHT_PX = 0

IMG_EXTS = (".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff")

def to_gray(img):
    # Carrega como BGR e converte para Gray; se já for 1 canal, apenas retorna
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def build_mask(h, w):
    mask = np.ones((h, w), dtype=bool)
    if EXCLUDE_TOP_RATIO > 0:
        mask[:int(h * EXCLUDE_TOP_RATIO), :] = False
    if EXCLUDE_BOTTOM_PX > 0:
        mask[h - EXCLUDE_BOTTOM_PX : , :] = False
    if EXCLUDE_LEFT_PX > 0:
        mask[:, :EXCLUDE_LEFT_PX] = False
    if EXCLUDE_RIGHT_PX > 0:
        mask[:, w - EXCLUDE_RIGHT_PX :] = False
    return mask

def estimate_global_lo_hi(filepaths):
    lows, highs = [], []
    for fp in tqdm(filepaths, desc="Passo 1/2: estimando lo/hi globais"):
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img is None: 
            continue
        g = to_gray(img).astype(np.float32)
        m = build_mask(*g.shape)
        vals = g[m]
        if vals.size < 100: 
            continue
        lo, hi = np.percentile(vals, [P_LO, P_HI])
        if hi <= lo: 
            hi = lo + 1.0
        lows.append(lo); highs.append(hi)
    # mediana é mais robusta do que média
    glo = float(np.median(lows)) if lows else 0.0
    ghi = float(np.median(highs)) if highs else 255.0
    if ghi <= glo: 
        ghi = glo + 1.0
    return glo, ghi

def normalize_gray(gray, lo, hi):
    gray = gray.astype(np.float32)
    norm = np.clip((gray - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    if CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        norm = clahe.apply(norm)
    return norm

def main():
    os.makedirs(DST, exist_ok=True)
    files = sorted(
        fp for fp in glob.glob(os.path.join(SRC, "**", "*"), recursive=True)
        if os.path.splitext(fp)[1].lower() in IMG_EXTS
    )
    if not files:
        print(f"Nenhuma imagem em {SRC}.")
        return

    meta = {"mode": MODE, "p_lo": P_LO, "p_hi": P_HI,
            "exclude": {"top_ratio": EXCLUDE_TOP_RATIO, "bottom_px": EXCLUDE_BOTTOM_PX,
                        "left_px": EXCLUDE_LEFT_PX, "right_px": EXCLUDE_RIGHT_PX},
            "clahe": CLAHE, "clahe_clip": CLAHE_CLIP, "clahe_tile": CLAHE_TILE,
            "images": []}

    if MODE == "dataset_fixed":
        glo, ghi = estimate_global_lo_hi(files)
        print(f"[GLOBAL] lo={glo:.2f} hi={ghi:.2f}")
        meta["global_lo"] = glo
        meta["global_hi"] = ghi
    else:
        glo = ghi = None

    for fp in tqdm(files, desc="Passo 2/2: normalizando e salvando"):
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img is None: 
            continue
        g = to_gray(img)
        h, w = g.shape
        if MODE == "per_frame":
            m = build_mask(h, w)
            vals = g[m].astype(np.float32)
            lo, hi = np.percentile(vals, [P_LO, P_HI])
            if hi <= lo: hi = lo + 1.0
        else:
            lo, hi = glo, ghi

        norm = normalize_gray(g, lo, hi)
        norm = cv2.resize(norm, (OUT_W, OUT_H), cv2.INTER_AREA)

        rel = os.path.relpath(fp, SRC)
        out_fp = os.path.join(DST, os.path.splitext(rel)[0] + ".png")
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        cv2.imwrite(out_fp, norm)

        meta["images"].append({
            "src": fp, "dst": out_fp, "lo": float(lo), "hi": float(hi),
            "orig_hw": [int(h), int(w)]
        })

    with open(os.path.join(DST, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"OK: {len(meta['images'])} frames normalizados em {DST}")

if __name__ == "__main__":
    main()

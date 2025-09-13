import cv2, os, glob, json, numpy as np

SRC = "dataset-greyscale"           # raiz com as imagens atuais (8-bit cinza do app)
DST = "dataset-greyscale-norm"      # saída padronizada
OUT_W, OUT_H = 512, 384
P_LO, P_HI = 5, 95
EXCLUDE_TOP_RATIO = 0.0             # 0.25 se houver “teto quente” recorrente

def normalize_gray(gray):
    gray = gray.astype(np.float32)
    h, w = gray.shape
    mask = np.ones_like(gray, dtype=bool)
    if EXCLUDE_TOP_RATIO > 0:
        mask[:int(h*EXCLUDE_TOP_RATIO), :] = False
    vals = gray[mask]
    lo, hi = np.percentile(vals, [P_LO, P_HI])
    if hi <= lo: hi = lo + 1.0
    norm = np.clip((gray - lo) * 255.0/(hi - lo), 0, 255).astype(np.uint8)
    return norm

def process_one(fp):
    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)  # já vem cinza
    norm = normalize_gray(img)
    norm = cv2.resize(norm, (OUT_W, OUT_H), cv2.INTER_AREA)
    return norm

def main():
    os.makedirs(DST, exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(SRC, "**", "*.jpg"), recursive=True) +
                  glob.glob(os.path.join(SRC, "**", "*.png"), recursive=True))
    meta = []
    for fp in imgs:
        out = process_one(fp)
        rel = os.path.relpath(fp, SRC)
        out_fp = os.path.join(DST, os.path.splitext(rel)[0] + ".png")
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        cv2.imwrite(out_fp, out)
        meta.append({"src": fp, "dst": out_fp})
    with open(os.path.join(DST, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"OK: {len(meta)} frames normalizados em {DST}")

if __name__ == "__main__":
    main()

# infer_anomaly.py
import os, glob, csv, cv2, numpy as np
from ultralytics import YOLO

# ---- CONFIG -----------------------------------------------------------------
IMG_DIR     = "../dataset-norm-2"                 # frames já normalizados (uint8 0..255)
MODEL       = "../runs/detect/train2/weights/best.pt"     # seu YOLO treinado
IMG_SIZE    = 512
CONF_TH     = 0.25                                      # confiança mínima do detector
TARGET_CLS  = {0}                                       # {0}=backpack, {1}=person, {0,1}=ambos

# heurísticas de anomalia térmica (ajuste no seu ambiente)
P95_THR     = 220                                       # exige cauda quente na ROI
DMEAN_THR   = 30                                        # exige ROI bem acima do fundo local
RING        = 8                                         # anel (px) ao redor da bbox para estimar fundo
ALERT_RULE  = 1.5                                       # soma dos critérios para ligar alerta

OUT_CSV     = "anomaly_events2.csv"
OUT_DIR     = "anomaly_viz2"
# -----------------------------------------------------------------------------

def anomaly_score(norm_u8: np.ndarray, xyxy, ring=RING,
                  p95_thr=P95_THR, dmean_thr=DMEAN_THR):
    """norm_u8: imagem em uint8 (0..255), xyxy: [x1,y1,x2,y2] em pixels"""
    x1, y1, x2, y2 = map(int, xyxy)
    H, W = norm_u8.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    roi = norm_u8[y1:y2, x1:x2]
    if roi.size < 25:
        return 0.0, {"p95": 0.0, "mean_roi": 0.0, "mean_bg": 0.0}

    p95 = float(np.percentile(roi, 95))
    mean_roi = float(roi.mean())

    bx1, by1 = max(0, x1 - ring), max(0, y1 - ring)
    bx2, by2 = min(W, x2 + ring), min(H, y2 + ring)
    bg = norm_u8[by1:by2, bx1:bx2].astype(np.float32)

    # mascarar a ROI para não “poluir” o fundo
    ry1, ry2 = y1 - by1, y2 - by1
    rx1, rx2 = x1 - bx1, x2 - bx1
    bg[ry1:ry2, rx1:rx2] = np.nan
    mean_bg = float(np.nanmean(bg)) if np.isfinite(bg).any() else 0.0

    score = (1.0 if p95 >= p95_thr else 0.0) + (1.0 if (mean_roi - mean_bg) > dmean_thr else 0.0)
    return score, {"p95": p95, "mean_roi": mean_roi, "mean_bg": mean_bg}

def to_rgb(im: np.ndarray) -> np.ndarray:
    """Ensure HxWx3 uint8 for YOLO."""
    if im is None:
        return None
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    elif im.ndim == 3 and im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    return np.clip(im, 0, 255).astype(np.uint8)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = YOLO(MODEL)

    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "**", "*.png"), recursive=True))
    if not imgs:
        print(f"[WARN] No images found under {IMG_DIR}")
        return

    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["img", "class", "conf", "x1","y1","x2","y2",
                     "p95","mean_roi","mean_bg","score","alert"])

        for fp in imgs:
            # leitura em cinza para score térmico
            gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"[WARN] Could not read {fp}")
                continue

            # conversão para RGB antes do YOLO
            rgb = to_rgb(gray)

            # inferência
            res = model.predict(source=rgb, imgsz=IMG_SIZE, conf=CONF_TH, verbose=False)[0]
            boxes = getattr(res, "boxes", None)

            relative_path = os.path.relpath(fp, IMG_DIR)
            unique_filename = relative_path.replace(os.path.sep, '_')
            out_path = os.path.join(OUT_DIR, unique_filename)

            if boxes is None or len(boxes) == 0:
                # ainda assim salvar um frame “limpo” para auditoria
                cv2.imwrite(out_path, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
                continue

            # base para overlay
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # tensores → numpy
            xyxy_all = boxes.xyxy.cpu().numpy()
            conf_all = boxes.conf.cpu().numpy()
            cls_all  = boxes.cls.cpu().numpy().astype(int)

            for xyxy, conf, cls in zip(xyxy_all, conf_all, cls_all):
                if cls not in TARGET_CLS:
                    continue

                score, stats = anomaly_score(gray, xyxy)
                alert = 1 if score >= ALERT_RULE else 0
                x1, y1, x2, y2 = map(int, xyxy)
                color = (0, 0, 255) if alert else (0, 255, 0)

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"c{cls} {conf:.2f} | p95={stats['p95']:.0f} μ={stats['mean_roi']:.0f} bg={stats['mean_bg']:.0f} s={score:.1f}"
                cv2.putText(vis, label, (x1, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

                wr.writerow([fp, cls, f"{conf:.3f}", x1, y1, x2, y2,
                             f"{stats['p95']:.1f}", f"{stats['mean_roi']:.1f}", f"{stats['mean_bg']:.1f}",
                             f"{score:.2f}", alert])

            cv2.imwrite(out_path, vis)

    print(f"[OK] CSV saved to {OUT_CSV}, visualizations in {OUT_DIR}/")

if __name__ == "__main__":
    main()

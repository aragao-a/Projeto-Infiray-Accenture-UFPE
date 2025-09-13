import os
import shutil
import random
from pathlib import Path

# ====== CONFIGURAÇÃO ======
SRC_DIR = Path("dataset-greyscale-norm")    # raiz das imagens normalizadas (.png/.jpg)
DST_DIR = Path("dataset-greyscale-yolo")    # saída YOLO
IMG_EXTS = {".png", ".jpg", ".jpeg"}        # extensões aceitas
TRAIN_RATIO = 0.80                           # 80/20 split
SEED = 42                                    # reprodutibilidade
# ==========================

def collect_images(root: Path):
    imgs = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    return sorted(imgs)

def prepare_dirs(dst: Path):
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (dst / sub).mkdir(parents=True, exist_ok=True)

def relative_no_ext(src_root: Path, img_path: Path) -> str:
    rel = img_path.relative_to(src_root).as_posix()
    noext = os.path.splitext(rel)[0]
    return noext

def copy_and_touch_label(src_root: Path, img_path: Path, dst_img_dir: Path, dst_lbl_dir: Path):
    rel_noext = relative_no_ext(src_root, img_path)
    # destino da imagem preservando subpastas
    dst_img_path = dst_img_dir / f"{rel_noext}.png"
    dst_img_path.parent.mkdir(parents=True, exist_ok=True)
    # copia como PNG (se já for PNG, copia; se for JPG, só copia – sem conversão)
    shutil.copy2(img_path, dst_img_path)

    # cria .txt vazio correspondente (negativo por padrão)
    dst_lbl_path = dst_lbl_dir / f"{rel_noext}.txt"
    dst_lbl_path.parent.mkdir(parents=True, exist_ok=True)
    if not dst_lbl_path.exists():
        dst_lbl_path.write_text("", encoding="utf-8")

def main():
    random.seed(SEED)
    imgs = collect_images(SRC_DIR)
    if not imgs:
        print(f"[ERRO] Nenhuma imagem encontrada em: {SRC_DIR.resolve()}")
        return

    print(f"[INFO] Imagens encontradas: {len(imgs)}")
    prepare_dirs(DST_DIR)

    # split
    random.shuffle(imgs)
    n_train = int(len(imgs) * TRAIN_RATIO)
    train_imgs = imgs[:n_train]
    val_imgs   = imgs[n_train:]

    print(f"[INFO] Split: train={len(train_imgs)}  val={len(val_imgs)}")

    # copiar + criar .txt vazio
    for i, p in enumerate(train_imgs, 1):
        copy_and_touch_label(SRC_DIR, p, DST_DIR / "images/train", DST_DIR / "labels/train")
        if i % 200 == 0:
            print(f"[train] {i}/{len(train_imgs)}")

    for i, p in enumerate(val_imgs, 1):
        copy_and_touch_label(SRC_DIR, p, DST_DIR / "images/val", DST_DIR / "labels/val")
        if i % 200 == 0:
            print(f"[val] {i}/{len(val_imgs)}")

    print(f"[OK] Dataset YOLO criado em: {DST_DIR.resolve()}")
    print("    Agora rotule as imagens com mochila editando os .txt correspondentes.")
    print("    Formato YOLO por linha: <class_id> <x_center> <y_center> <width> <height> (normalizado 0..1)")

   

if __name__ == "__main__":
    main()

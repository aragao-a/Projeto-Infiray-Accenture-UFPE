import os
import shutil
import random
from pathlib import Path

# ====== CONFIGURAÇÃO ======
SRC_DIR = Path("dataset-norm-2")            # raiz das imagens normalizadas (.png/.jpg)
DST_DIR = Path("dataset-yolo-classified")   # saída YOLO
IMG_EXTS = {".png", ".jpg", ".jpeg"}        # extensões aceitas
TRAIN_RATIO = 0.80                           # 80/20 split
SEED = 42                                    # reprodutibilidade
CLASS_MAP = {"backpack": 0, "person": 1}     # mapeamento de classes
# ==========================

def collect_images(root: Path):
    imgs = []
    for class_name in CLASS_MAP.keys():
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                imgs.append(p)
    return sorted(imgs)

def prepare_dirs(dst: Path):
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (dst / sub).mkdir(parents=True, exist_ok=True)

def relative_no_ext(src_root: Path, img_path: Path) -> str:
    # We want the path relative to the class folder, not the root, to avoid creating class subdirs in the output
    rel = img_path.relative_to(img_path.parent.parent).as_posix()
    noext = os.path.splitext(rel)[0]
    return noext

def copy_and_create_label(src_root: Path, img_path: Path, dst_img_dir: Path, dst_lbl_dir: Path):
    class_name = img_path.parent.name
    if class_name not in CLASS_MAP:
        return 

    class_id = CLASS_MAP[class_name]
    
    rel_noext = relative_no_ext(src_root, img_path)

    # destino da imagem
    dst_img_path = dst_img_dir / f"{rel_noext}.png"
    dst_img_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, dst_img_path)

    # cria .txt com a classe e bounding box de imagem inteira
    dst_lbl_path = dst_lbl_dir / f"{rel_noext}.txt"
    dst_lbl_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Bounding box cobrindo a imagem inteira
    yolo_bbox = f"{class_id} 0.5 0.5 1.0 1.0"
    dst_lbl_path.write_text(yolo_bbox, encoding="utf-8")

def create_data_yaml(dst_dir: Path):
    content = f"""
path: {dst_dir.resolve()}
train: images/train
val: images/val
names:
  0: backpack
  1: person
"""
    (dst_dir / "data.yaml").write_text(content.strip())

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

    # copiar + criar .txt
    for i, p in enumerate(train_imgs, 1):
        copy_and_create_label(SRC_DIR, p, DST_DIR / "images/train", DST_DIR / "labels/train")
        if i % 200 == 0:
            print(f"[train] {i}/{len(train_imgs)}")

    for i, p in enumerate(val_imgs, 1):
        copy_and_create_label(SRC_DIR, p, DST_DIR / "images/val", DST_DIR / "labels/val")
        if i % 200 == 0:
            print(f"[val] {i}/{len(val_imgs)}")

    create_data_yaml(DST_DIR)

    print(f"[OK] Dataset YOLO criado em: {DST_DIR.resolve()}")

if __name__ == "__main__":
    main()

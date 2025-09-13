# Infiray P2 PRO - Câmera Térmica com IA

**Kickstart do Projeto - Fase 1 da Rotação Accenture UFPE**

Projeto da Câmera Térmica do Innovation Center com objetivo inicial de captação eficiente do feed da câmera para utilização em modelos de IA e aplicações industriais.

## 📌 Pipeline da Fase 1

### 1. Aquisição e recorte dos frames
- Extração de quadros da câmera térmica via aplicativo **P2Pro**.
- Corte das regiões de interesse para eliminar bordas e sobreposições da interface.

### 2. Normalização por percentis (p5–p95)
- **Frame a frame**: pega o histograma de intensidades e descarta os 5% mais baixos e os 5% mais altos.
- O intervalo central (90%) é remapeado para [0–255].
- Isso aumenta o contraste interno de cada frame.
- ➝ **Resultado**: dataset consistente para identificar anomalias locais de temperatura.
- **Script usado**: `prep_norm.py` → gera `dataset-greyscale-norm`.

### 3. Criação de labels
- **Ferramenta**: MakeSense.AI (mais estável que LabelImg no Windows).
- **Classes definidas**:
  - `0` → backpack (mochila)
  - `1` → person (pessoa)
- **Formato YOLO**: `.txt` com `<class> <x_center> <y_center> <width> <height>` (valores normalizados 0–1).

### 4. Estrutura do dataset YOLO
- `images/train/` e `images/val/` → imagens.
- `labels/train/` e `labels/val/` → anotações `.txt`.
- Arquivo `data.yaml` definindo paths e classes.

### 5. Treinamento YOLOv8
- **Modelo base**: `yolov8n.pt` (nano, rápido para protótipo).
- **Comando usado**:
  ```bash
  yolo detect train model=yolov8n.pt data=dataset-greyscale-yolo/data.yaml imgsz=512 epochs=50 batch=16
# Infiray P2 PRO - Câmera Térmica com IA
## Kickstart do Projeto - Fase 1 da Rotação Accenture UFPE
Projeto da Câmera Térmica do Innovation Center com objetivo inicial de captação eficiente do feed da câmera pra utilização em modelos de IA e aplicações industriais.
## Registros de Encontros no Innovation Center
### 1° Encontro: 31/08/2025
- No primeiro contato com a câmera, exploramos o ambiente e subimos uma pasta no Drive com materiais de captação da câmera pra estudo e uso posterior (https://drive.google.com/drive/folders/17xsEMq_SosNN9f8nSo6HjP_qbQsIrgn9?usp=drive_link)
- Debatemos com Camilla possibilidades de integração com o cachorro, que sugeriu o feed alternativo no Dashboard dele- Por fim, a decisão consciente foi atacar primeiro o problema de como obter o feed da câmera de forma eficiente, com avenidas de oportunidades imediatas sendo:    
    - A transmissão do celular Android pra notebooks usando o ADB e tratando o feed a nível de exibição visual no Notebook, tipo rodar um Streaming    
    - Explorar capacidades de IP ou MAC Address do aplicativo P2 PRO    
    - Explorar repositório do GitHub existente que se propõe a isso - https://github.com/ks00x/p2pro-live
### Reunião Recorrente com André (11/08): 
- Veredito que o celular vai ter que ser uma espécie de middleware para realizar a transmissão dos conteúdos
- Conferir com André Aragão a ideia adotada na transmissão do Rayban
- Conferir legalidade dos processos envolvidos pra Transmissão 
- Teste de transmissão da tela lá no IC
- Teste da Hipótese OpenCV
### Reunião Recorrente com André (18/08): 
- Parte da equipe foi para o Innovation Center testar as possibilidades de conexão da câmera com o computador.
- André incentivou a ideia e o avanço pode tornar promissor a possibilidade de integração com o cachorro robô como uma feature futura.
### Reunião Recorrente com André (25/08): 
- Avaliar a possibilidade de deixar um range específico.
- André pediu para verificar a questão de dinamicidade da câmera em relação ao meio.
### 2° Encontro (18/08): 
- 1° Alternativa testada e validada - Transmissão da Câmera diretamente no Desktop, via App Oficial e reconhecido como Webcam ✅
- 2° Alternativa testada e validada - Transmissão wireless Celular -> Notebook via SCRCPY, mais flexível e congruente com a aplicação Unitree GO1 ✅
- Comandos importantes via SCRCPY:
```bash
    adb devices
    adb tcpip 5555
    (remover cabo)
    adb connect {ip}:5555 (no mesmo wi-fi)
    adb devices (confirmar conexão wireless)
    scrcpy

```
### Colabs de Experimentação pra obtenção das Temperaturas / Treinamento de Modelo: 
- https://colab.research.google.com/drive/1kqmdpX7kIXMH38yqwNZwp9BbsJlA-Wsn?usp=sharing
- https://colab.research.google.com/drive/11X_l1v3IsetNd6W9OgqWgKdKm6aNOc2m?usp=sharing
## Ideias de Inovação pós critérios de aceite
### Aplicação no Unitree GO1
- Achar uma forma de acoplar o feed da Câmera ao Dashboard do cachorro Unitree GO1, oferecendo uma nova camada de visão, incluindo leituras visuais térmicos no escuro e de equipamentos industriais ou em ambientes tecnológicos
- Ademais, a câmera pode servir como um guia de foco rápido a pontos de interesse pro cachorro
### Aplicação em salas de reunião
- Estabelecer como uma câmera fixa que avalia ambientes contando quantas pessoas estão ali e associando com horários
### Detector de Invasão
- Pela capacidade de visão no escuro, boa pra varrer salas sem forte incidência de luz

# Conclusões gerais:

## Modelos:

- Treinamos Yolo V8 SON, Yolo SPP piorou, mas retornamos ao motivo que o Dataset é limitado e o ambiente pra obter ele também, difícil criar situações reais.
- Apesar de tudo, fizemos uma pipeline de bons resultados que compreende: Script de inferência que recebe as imagens reconhecidas como bolsas (a partir de um Modelo), e analisa se é quente a partir do gradiente de cor.

## Integração Feed Câmera - API Dados Brutos:

- Jetson funcionou com a câmera, mas o Link com o cachorro não foi possível de forma alguma, devido a interação Bluetooth (Jetson não tem) e rede IOT.
- Link câmera-cachorro também não foi possível pois a Raspberry do cachorro não reconhece a câmera.
- Porém como a Rasp dele está conectada à IOT da Acc, seria interessante a conexão pra pegar essas infos.

# Treinamentos e Produções: Infiray P2 Pro - AI Thermal Camera Project

**Project from the Accenture-UFPE Innovation Center Rotation**

This project provides a complete pipeline for using the InfiRay P2 Pro thermal camera to train and deploy a YOLOv8 object detection model for anomaly detection. The initial goal is to efficiently capture the camera's feed for use in AI models and industrial applications.

---

## Features

- **Real-time Anomaly Detection:** Detects `people` and `backpacks` from a thermal camera feed.
- **Complete ML Pipeline:** Scripts for data preparation, training, and inference.
- **Extensible:** Built with YOLOv8, easy to train on new custom objects.

---

## Project Structure

The project is organized to separate code, data, and results for clarity and ease of use.

```
/
├── .gitignore                # Files to ignore in git
├── README.md                 # This guide
├── requirements.txt          # Project dependencies
├── data.yaml                 # YOLO dataset configuration
├── yolov8n.pt                # Base model for reproducibility
│
├── src/                      # All project source code
│   ├── data_preparation/     # Scripts to process raw data
│   ├── inference.py          # Script to run detection on a file
|   ├── p2prolive_app.py      # Script to run live detection with the camera
|   └── thermal_detector.py   # Module which integrates the model with the thermal anomaly verification algorithm
│
├── dataset/                  # Raw image data (not in git)
│
└── runs/                     # YOLO training and detection outputs (not in git)
```

---

## Setup and Installation

Follow these steps to set up your local development environment.

1.  **Prerequisites:**
    *   Python 3.8+
    *   Git

2.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

3.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Use the Pipeline

### 1. Data Preparation

Before training, raw images must be processed into the YOLO format.

-   Place your raw images into the `dataset/` folder, sorted by class (e.g., `dataset/person/`, `dataset/backpack/`).
-   Run the data preparation scripts located in `src/data_preparation/` to normalize the images and create the training/validation splits with `.txt` labels.
    *Note: The exact commands for these scripts should be documented here.*

### 2. Model Training

To train the model, use the `yolo` command-line interface. The following command trains a `yolov8n` model for 100 epochs with data augmentation and early stopping.

-   **Recommended command:**
    ```bash
    yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 patience=20 batch=16 imgsz=512 augment=false
    ```
-   Training results, including the best model weights (`best.pt`), will be saved in a new directory under `runs/detect/` (e.g., `runs/detect/train1`).

### 3. Inference

Once the model is trained, you can use it for detection.

-   **To run inference on a video file:**
    ```bash
    python src/inference.py --weights runs/detect/train1/weights/best.pt --source path/to/your/video.mp4
    ```

-   **To run live inference with the InfiRay P2 Pro camera:**
    ```bash
    streamlit run p2prolive_app.py
    ```
    *(Note: Command-line arguments for inference scripts may need to be adjusted.)*

---

## Training Analysis

This section provides an overview of the different training runs performed for this project.

### Train 1 (`runs/detect/train`)

This training was performed using the following parameters:

*   **Model:** `yolov8s.pt`
*   **Dataset:** `dataset-yolo-classified/data.yaml`
*   **Epochs:** 100 (stopped at 23 due to patience)
*   **Patience:** 20
*   **Image Size:** 512
*   **Batch Size:** 16

The training stopped early at epoch 23. The best results were:
*   **mAP50(B):** 0.73368
*   **mAP50-95(B):** 0.3951

### Train 2 (`runs/detect/train2`)

This training was performed with a different dataset configuration.

*   **Model:** `yolov8s.pt`
*   **Dataset:** `data.yaml`
*   **Epochs:** 100 (stopped at 4 due to patience)
*   **Patience:** 20
*   **Image Size:** 512
*   **Batch Size:** 16

The training stopped very early at epoch 4. The best results were:
*   **mAP50(B):** 0.73368
*   **mAP50-95(B):** 0.3951

### Train 4 (`runs/detect/train4`)

This training used the `yolov8n.pt` model and ran for the full 50 epochs.

*   **Model:** `yolov8n.pt`
*   **Dataset:** `dataset-yolo-classified/data.yaml`
*   **Epochs:** 50
*   **Patience:** 100
*   **Image Size:** 512
*   **Batch Size:** 16

The training completed all 50 epochs. The best results were:
*   **mAP50(B):** 0.91487
*   **mAP50-95(B):** 0.5039

#### Results for Train 4

![Results](runs/detect/train4/results.png)
![Confusion Matrix](runs/detect/train4/confusion_matrix_normalized.png)

---

## Contributing

Contributions are welcome! Please follow these steps:

1.  **Fork** the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  **Commit** your changes (`git commit -m 'Add some feature'`).
5.  **Push** to the branch (`git push origin feature/your-feature-name`).
6.  Open a **Pull Request**.

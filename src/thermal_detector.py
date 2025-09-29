# thermal_detector.py
import cv2
import numpy as np
from ultralytics import YOLO
import torch

class ThermalDetector:
    def __init__(self, model_path, thermal_config=None):
        """
        Inicializa o detector térmico com algoritmo original do Infiray
        
        Args:
            model_path: caminho para o modelo YOLOv8 treinado
            thermal_config: configurações do algoritmo de anomalia
        """
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        
        self.config = thermal_config or {
            'P95_THR': 220,        # percentil 95 threshold
            'DMEAN_THR': 30,       # diferença média threshold  
            'RING': 8,             # anel para fundo
            'ALERT_RULE': 1.5,     # regra de alerta
            'CONF_TH': 0.25,       # confiança mínima YOLO
            'TARGET_CLS': {0},     # classes alvo (0=backpack)
            'IMG_SIZE': 512        # tamanho da imagem
        }
        
    def anomaly_score(self, norm_u8: np.ndarray, xyxy):
        """
        ALGORITMO ORIGINAL DO INFIRAY - DETECÇÃO DE ANOMALIA TÉRMICA
        norm_u8: imagem em uint8 (0..255), xyxy: [x1,y1,x2,y2] em pixels
        """
        x1, y1, x2, y2 = map(int, xyxy)
        H, W = norm_u8.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        roi = norm_u8[y1:y2, x1:x2]
        if roi.size < 25:
            return 0.0, {"p95": 0.0, "mean_roi": 0.0, "mean_bg": 0.0}

        p95 = float(np.percentile(roi, 95))
        mean_roi = float(roi.mean())

        # Cálculo do fundo local
        bx1, by1 = max(0, x1 - self.config['RING']), max(0, y1 - self.config['RING'])
        bx2, by2 = min(W, x2 + self.config['RING']), min(H, y2 + self.config['RING'])
        bg = norm_u8[by1:by2, bx1:bx2].astype(np.float32)

        # Mascarar a ROI para não "poluir" o fundo
        ry1, ry2 = y1 - by1, y2 - by1
        rx1, rx2 = x1 - bx1, x2 - bx1
        bg[ry1:ry2, rx1:rx2] = np.nan
        mean_bg = float(np.nanmean(bg)) if np.isfinite(bg).any() else 0.0

        # Cálculo do score
        score = (1.0 if p95 >= self.config['P95_THR'] else 0.0) + \
                (1.0 if (mean_roi - mean_bg) > self.config['DMEAN_THR'] else 0.0)
        
        return score, {"p95": p95, "mean_roi": mean_roi, "mean_bg": mean_bg}

    def to_rgb(self, im: np.ndarray) -> np.ndarray:
        """Ensure HxWx3 uint8 for YOLO - função original do Infiray"""
        if im is None:
            return None
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        elif im.ndim == 3 and im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        return np.clip(im, 0, 255).astype(np.uint8)

    def detect_objects(self, rgb_image, thermal_data=None):
        """
        Detecta objetos e aplica algoritmo de anomalia térmica original
        
        Args:
            rgb_image: imagem RGB para detecção
            thermal_data: dados térmicos em uint8 (0-255) - IMPORTANTE!
            
        Returns:
            resultados com detecções e anomalias
        """
        # Executa detecção YOLOv8
        results = self.model(rgb_image, 
                           imgsz=self.config['IMG_SIZE'], 
                           conf=self.config['CONF_TH'], 
                           verbose=False)
        
        detections = []
        thermal_anomalies = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Informações da detecção
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[cls]
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_name': class_name,
                        'class_id': cls
                    }
                    
                    # VERIFICA ANOMALIA TÉRMICA APENAS PARA CLASSES ALVO
                    if (cls in self.config['TARGET_CLS'] and 
                        thermal_data is not None and 
                        thermal_data.dtype == np.uint8):
                        
                        score, thermal_stats = self.anomaly_score(thermal_data, [x1, y1, x2, y2])
                        alert = score >= self.config['ALERT_RULE']
                        
                        detection['thermal_anomaly'] = {
                            'has_anomaly': alert,
                            'score': score,
                            'p95': thermal_stats['p95'],
                            'mean_roi': thermal_stats['mean_roi'],
                            'mean_bg': thermal_stats['mean_bg'],
                            'thresholds': {
                                'p95_thr': self.config['P95_THR'],
                                'dmean_thr': self.config['DMEAN_THR'],
                                'alert_rule': self.config['ALERT_RULE']
                            }
                        }
                        
                        if alert:
                            thermal_anomalies.append({
                                'bbox': [x1, y1, x2, y2],
                                'class_name': class_name,
                                'score': score,
                                'thermal_stats': thermal_stats
                            })
                    else:
                        detection['thermal_anomaly'] = {
                            'has_anomaly': False,
                            'score': 0.0
                        }
                    
                    detections.append(detection)
        
        return {
            'detections': detections,
            'thermal_anomalies': thermal_anomalies,
            'original_results': results
        }
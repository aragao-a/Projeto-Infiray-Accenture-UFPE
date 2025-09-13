# Infiray P2 PRO - Câmera Térmica com IA
## Kickstart do Projeto - Fase 1 da Rotação Accenture UFPE
Projeto da Câmera Térmica do Innovation Center com objetivo inicial de captação eficiente do feed da câmera pra utilização em modelos de IA e aplicações industriais.
## Registros de Encontros no Innovation Center
1. Normalização por percentis (p5–p95)

Quando você pega imagens da câmera, os valores de pixel (intensidade) representam temperaturas mapeadas pela paleta do app. Só que isso varia muito de cena pra cena — em um frame pode ir de 20°C a 40°C, em outro de 10°C a 25°C.

Normalização por frame (p5–p95):
Para cada frame, você pega o histograma de intensidades e corta os 5% mais baixos e os 5% mais altos. O que sobra (90% central) é remapeado para a escala de cinza [0–255].
➝ Resultado: cada frame fica “otimizado” para o contraste interno dele. Bom pra destacar anomalias locais, mas ruim se você quer consistência global (comparar frames de momentos diferentes).

Normalização fixa por cena:
Você define os valores mínimo e máximo a partir de toda a sequência (por exemplo: min = 20°C, máx = 40°C) e aplica o mesmo mapeamento em todos os frames.
➝ Resultado: perde contraste em algumas imagens, mas permite comparabilidade entre frames, essencial se você vai identificar ranges de temperatura fixos.

No caso (detecção de anomalias térmicas), a normalização fixa por cena ou por dataset é melhor.

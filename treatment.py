import cv2
import os
import glob
import shutil

pasta_saida = 'dataset-color'
intervalo_segundos = 2
extensao_video = '.mkv'

pixels_remover_topo = 98
pixels_remover_base = 210
pixels_remover_esquerda = 442
pixels_remover_direita = 442

caminho_pesquisa = os.path.join(os.getcwd(), f'*{extensao_video}')
lista_de_videos = glob.glob(caminho_pesquisa)

if not lista_de_videos:
    print(f"Nenhum arquivo de vídeo '{extensao_video}' encontrado na pasta.")
    exit()

print(f"Encontrados {len(lista_de_videos)} vídeos para processar:")
for video in lista_de_videos:
    print(f" - {os.path.basename(video)}")
print("-" * 30)

if os.path.exists(pasta_saida):
    shutil.rmtree(pasta_saida)
os.makedirs(pasta_saida)

for caminho_do_video in lista_de_videos:
    nome_base_video = os.path.splitext(os.path.basename(caminho_do_video))[0]
    print(f"\nProcessando vídeo: {nome_base_video}{extensao_video}")
    
    cap = cv2.VideoCapture(caminho_do_video)

    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
        
    intervalo_frames = int(fps * intervalo_segundos)
    contador_frames_total = 0

    while True:
        sucesso, frame = cap.read()
        if not sucesso:
            break

        if contador_frames_total % intervalo_frames == 0:
            altura_frame, largura_frame, _ = frame.shape

            y_start = pixels_remover_topo
            y_end = altura_frame - pixels_remover_base
            
            x_start = pixels_remover_esquerda
            x_end = largura_frame - pixels_remover_direita

            frame_cortado = frame[y_start:y_end, x_start:x_end]

            nome_screenshot = f"{nome_base_video}_frame_{contador_frames_total}.jpg"
            caminho_salvar = os.path.join(pasta_saida, nome_screenshot)

            cv2.imwrite(caminho_salvar, frame_cortado)

        contador_frames_total += 1

    cap.release()
    print(f"Processamento de {nome_base_video}{extensao_video} concluído.")

print("\nTodos os vídeos foram processados")
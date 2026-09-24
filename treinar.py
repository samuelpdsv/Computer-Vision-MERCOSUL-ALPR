from ultralytics import YOLO

def main():
    print("Iniciando o Treinamento do YOLOv8...")
    
    model = YOLO("yolov8n.pt")

    
    resultados = model.train(
        data="dataset/data.yaml", # Caminho do arquivo de configuração
        epochs=50,                # vezes que ele vai ver o dataset inteiro 
        imgsz=640,                # Tamanho de redimensionadas de imagens para treino
        batch=8,                  # Quantidade de imagens processadas por vez pela GPU
        device=0,                 # força o uso da Placa de Vídeo NVIDIA
        plots=True,               # Gera gráficos
        project="Treinamento",    # Nome da pasta dos resultados
        name="Placas_Mercosul"    # Nome da subpasta dos resultados
    )
    
    print("Treinamento Concluído com Sucesso!")

if __name__ == '__main__':
    main()
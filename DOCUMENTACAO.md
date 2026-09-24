# Sistema de Reconhecimento de Placas - Documentação Completa

## Visão Geral
Sistema de reconhecimento automático de placas de veículos no padrão Mercosul (Brasil), utilizando YOLOv8 para detecção de objetos e Tesseract OCR para reconhecimento de caracteres, com validação lógica de formato.

## Arquitetura do Sistema

### 1. Detecção de Placas (YOLOv8)
- Modelo trein com dataset de 472 imagens de placas Mercosul
- Arquivo de treinamento: `treinar.py`
- Modelo final: `runs/detect/Treinamento/Placas_Mercosul4/weights/best.pt`
- Detecta a região da placa dentro da imagem/cenário

### 2. Recorte Inteligente
- Remove 25% superior da região (faixa azul da placa)
- Remove 5% laterais (bordas pretas)
- Realiza ampliação 2x e binarização Otsu para melhorar a leitura OCR

### 3. OCR (Tesseract)
- Configuração PSM 7 (única linha de texto)
- Whitelist de caracteres: A-Z, 0-9
- Correção de erros comuns de leitura

### 4. Validação e Correção de Texto
Função `corrigir_placa_mercosul()`:
- Remove caracteres alfanuméricos apenas
- Aplica correções de caracteres confusos (0=O, 1=I, 2=Z, etc.)
- Valida formato: `^[A-Z]{3}[0-9][A-Z][0-9]{2}$`
- Formato Mercosul: AAA-NNN-AA (3 letras, 1 número, 1 letra, 2 números)

## Arquivos Principais

| Arquivo | Descrição |
|---------|-----------|
| `reconhecer.py` | Execução única em imagem estática (`carro6.jpg`) |
| `video_reconhecer.py` | Reconhecimento em tempo real de webcam/video |
| `treinar.py` | Treinamento do modelo YOLOv8 |
| `verificar.py` | Verificação da estrutura do dataset |
| `teste.py` | Teste de ambiente (OpenCV, Tesseract, YOLO, GPU) |
| `corrigir_placa_mercosul()` | Função de validação e correção de texto |

## Como Usar

### Reconhecimento em Imagem
```bash
python reconhecer.py
```

### Reconhecimento em Vídeo/Webcam
```bash
python video_reconhecer.py
```

### Treinamento do Modelo
```bash
python treinar.py
```
*(Requiere dataset no `dataset/` e GPU NVIDIA)*

## Pré-requisitos
- Python 3.8+
- OpenCV (`pip install opencv-python`)
- Tesseract OCR instalado no Windows com path `C:\Program Files\Tesseract-ocr\tesseract.exe`
- PyTesseract (`pip install pytesseract`)
- Ultralytics YOLO (`pip install ultralytics`)
- PyTorch (`pip install torch`)
- GPU NVIDIA recomendada para treinamento

## Limitações
- Apenas placas no padrão Mercosul (AAA-NNN-AA)
- Requer iluminação adequada da placa
- Modelo treinado com dataset específico pode não funcionar com placas de outros estados/padrões
- Correção baseada em dicionário de erros comuns - pode falhar em casos exóticos

## Resultados Esperados
- Detecção de placa com bounding box verde
- Texto da placa exibido acima da caixa
- Console mostra "PLACA VALIDADA: [PLACA]" quando reconhecido com sucesso
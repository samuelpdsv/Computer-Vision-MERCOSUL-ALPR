import cv2
import pytesseract
from ultralytics import YOLO
import torch

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("--- RELATÓRIO DO AMBIENTE ---")
print("1. OpenCV versão:", cv2.__version__)
print("2. Tesseract configurado no caminho:", pytesseract.pytesseract.tesseract_cmd)
print("3. YOLO importado com sucesso!")
print("4. Placa de Vídeo (GPU) detectada pelo PyTorch?:", torch.cuda.is_available())
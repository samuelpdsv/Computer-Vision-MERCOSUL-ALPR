import cv2
import pytesseract
import re  #Regex  expressões regulares
from ultralytics import YOLO

# --- FUNÇÃO DE INTELIGÊNCIA DO OCR ---
def corrigir_placa_mercosul(texto_lido):
    texto = "".join(e for e in texto_lido if e.isalnum()).upper()
    
    if len(texto) < 7:
        return None
        
    if len(texto) > 7:
        texto = texto[-7:]
        
    #Dicionários de correção para as confusões mais comuns do Tesseract
    letras_erradas = {'0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A', '5': 'S', '6': 'G', '7': 'Z', '8': 'B'}
    numeros_errados = {'O': '0', 'I': '1', 'Z': '2', 'B': '8', 'A': '4', 'S': '5', 'G': '6', 'T': '7', 'U': '0', 'C': '0', 'D': '0'}
    
    placa_corrigida = ""
    
    for i, char in enumerate(texto):
        if i in [0, 1, 2, 4]: 
            placa_corrigida += letras_erradas.get(char, char)
        else:                  
            placa_corrigida += numeros_errados.get(char, char)
            
    if re.match(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$', placa_corrigida):
        return placa_corrigida
    else:
        return None

# 1. Configurações
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
caminho_modelo = r'runs\detect\Treinamento\Placas_Mercosul4\weights\best.pt' # Atualize para o caminho do seu modelo treinado
modelo = YOLO(caminho_modelo)

fonte_video = 0  
video = cv2.VideoCapture(fonte_video)

contador_frame = 0

print(" Iniciando a captura... (Aperte 'x' na janela do vídeo para sair)")

while True:
    sucesso, frame = video.read()
    if not sucesso:
        break
        
    contador_frame += 1 

    resultados = modelo(frame, verbose=False)

    for resultado in resultados:
        for caixa in resultado.boxes:
            x1, y1, x2, y2 = map(int, caixa.xyxy[0])
            if y1 < 0 or y2 > frame.shape[0] or x1 < 0 or x2 > frame.shape[1]:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #SMART CROP
            altura = y2 - y1
            largura = x2 - x1
            
            corte_topo = int(altura * 0.25) # Tira 25% de cima (Faixa Azul)
            corte_lateral = int(largura * 0.05) # Tira 5% de cada lado (Bordas Pretas)
            
            placa_recortada = frame[y1 + corte_topo : y2, x1 + corte_lateral : x2 - corte_lateral]

            if placa_recortada.size != 0:
                cinza = cv2.cvtColor(placa_recortada, cv2.COLOR_BGR2GRAY)
                _, binarizada = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                texto_sujo = pytesseract.image_to_string(binarizada, config='--psm 8')
                placa_validada = corrigir_placa_mercosul(texto_sujo)

                if placa_validada:
                    cv2.putText(frame, placa_validada, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                     
                    print(f"[ Frame {contador_frame} ] Placa identificada: {placa_validada}")

    cv2.imshow("Reconhecimento Inteligente", frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

video.release()
cv2.destroyAllWindows()
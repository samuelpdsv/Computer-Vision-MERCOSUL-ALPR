import cv2
import pytesseract
import re
from ultralytics import YOLO

# --- FUNÇÃO DE INTELIGÊNCIA DO OCR ---
def corrigir_placa_mercosul(texto_lido):
    texto = "".join(e for e in texto_lido if e.isalnum()).upper()
    
    if len(texto) < 7: return None
    if len(texto) > 7: texto = texto[-7:]
        
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
    return None

# Configurar o caminho do Tesseract 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

caminho_modelo = r'runs\detect\Treinamento\Placas_Mercosul4\weights\best.pt'
modelo = YOLO(caminho_modelo)

# Carregar a imagem de teste
imagem = cv2.imread('carro6.jpg')

if imagem is None:
    print("Erro: Não achei a imagem 'carro.jpg'. Verifique o nome e se ela está na pasta certa!")
else:
    print("Analisando a imagem...")
    resultados = modelo(imagem)

    for resultado in resultados:
        for caixa in resultado.boxes:
            x1, y1, x2, y2 = map(int, caixa.xyxy[0])
            
            cv2.rectangle(imagem, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # --- SMART CROP---
            altura = y2 - y1
            largura = x2 - x1
            
            corte_topo = int(altura * 0.25) 
            corte_esq = int(largura * 0.05) # Corta 5% da esquerda (mata o "L")
            corte_dir = int(largura * 0.01) # Corta só 1% da direita (protege o "4")
            
            placa_recortada = imagem[y1 + corte_topo : y2, x1 + corte_esq : x2 - corte_dir]

            # --- PIPELINE DE IMAGEM: LIMPO E COM BORDA (PADDING) ---
            if placa_recortada.size != 0:
                # 1. Amplia 2x
                placa_ampliada = cv2.resize(placa_recortada, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                
                # 2. Tons de Cinza
                cinza = cv2.cvtColor(placa_ampliada, cv2.COLOR_BGR2GRAY)
                
                # 3. Binarização de Otsu
                _, binarizada = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                
                # 4. Adiciona 15 pixels de borda branca ao redor de tudo
                binarizada_final = cv2.copyMakeBorder(binarizada, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])

                cv2.imshow("Placa Pronta para o Tesseract", binarizada_final)

                # 5. Leitura Tesseract (PSM 7 e Whitelist)
                config_tess = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                texto_sujo = pytesseract.image_to_string(binarizada_final, config=config_tess)
                
                # 6. Motor Lógico
                placa_validada = corrigir_placa_mercosul(texto_sujo)

                if placa_validada:
                    print(f"PLACA VALIDADA: {placa_validada}")
                    cv2.putText(imagem, placa_validada, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                else:
                    print(f"Placa detectada, mas a leitura falhou: {texto_sujo.strip()}")

            cv2.imshow("Placa Recortada (OpenCV)", placa_recortada)
    
    cv2.imshow("Imagem Completa (YOLO)", imagem)
    
    print("Pressione qualquer tecla na janela da imagem para fechar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
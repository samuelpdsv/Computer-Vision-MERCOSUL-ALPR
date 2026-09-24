import os

caminho = r"C:\Users\samue\Desktop\Reconhecimento_Placas\dataset"
print("Varredura de pastas")

if os.path.exists(caminho):
    itens = os.listdir(caminho)
    print(f"Encontrado dentro de 'dataset': {itens}")
    
    for item in itens:
        caminho_interno = os.path.join(caminho, item)
        if os.path.isdir(caminho_interno):
            print(f"  -> Dentro de '{item}' tem: {os.listdir(caminho_interno)[:5]} (mostrando até 5 itens...)")
else:
    print("A pasta 'dataset' principal não existe nesse local!")
# criar arquivo: comprimir_modelo.py
import joblib
import gzip
import pickle

print("📦 Comprimindo modelo...")

# Carregar modelo original
modelo = joblib.load('modelo_clima_joao_pessoa.pkl')

# Salvar comprimido
with gzip.open('modelo_clima_joao_pessoa.pkl.gz', 'wb') as f:
    pickle.dump(modelo, f)

print("✅ Modelo comprimido!")

import os
tamanho_original = os.path.getsize('modelo_clima_joao_pessoa.pkl') / 1024 / 1024
tamanho_comprimido = os.path.getsize('modelo_clima_joao_pessoa.pkl.gz') / 1024 / 1024

print(f"Original: {tamanho_original:.2f} MB")
print(f"Comprimido: {tamanho_comprimido:.2f} MB")
print(f"Redução: {(1 - tamanho_comprimido/tamanho_original)*100:.1f}%")
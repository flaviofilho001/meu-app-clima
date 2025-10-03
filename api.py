# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import os
import gzip
import pickle

# =====================================================
# CONFIGURAÇÕES
# =====================================================

# Caminhos dos arquivos
MODEL_PATH = 'modelo_clima_joao_pessoa.pkl.gz'
DATA_PATH = 'dados_capitais.csv'

# Verificar se arquivos existem
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dados não encontrados: {DATA_PATH}")

# Carregar modelo
print("📦 Carregando modelo...")
with gzip.open('modelo_clima_joao_pessoa.pkl.gz', 'rb') as f:
    modelo = pickle.load(f)
print(f"✅ Modelo carregado: {modelo['cidade']}")

# Carregar dados históricos
print("📊 Carregando dados históricos...")
df_historico = pd.read_csv(DATA_PATH)
df_historico = df_historico.replace(-999.0, np.nan)
df_historico = df_historico.replace(-3596.4, np.nan)
df_historico['Data'] = pd.to_datetime(df_historico['Data'].astype(str), format='%Y%m%d', errors='coerce')
df_historico = df_historico[df_historico['Cidade'] == modelo['cidade']].sort_values('Data')
df_historico = df_historico.dropna(subset=['Temp Média (°C)'])
print(f"✅ Dados carregados: {len(df_historico)} registros")

# =====================================================
# CRIAR API
# =====================================================

app = FastAPI(
    title="🌤️ API de Previsão Climática",
    description="API para prever condições meteorológicas usando ML",
    version="1.0.0"
)

# CORS - Permitir acesso de qualquer origem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# MODELOS DE DADOS
# =====================================================

class PrevisaoRequest(BaseModel):
    data: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": "2025-10-04"
            }
        }

class PrevisaoResponse(BaseModel):
    data: str
    cidade: str
    dia_semana: str
    temperatura_media: float
    temperatura_maxima: float
    temperatura_minima: float
    umidade: float
    precipitacao: float
    vento: float
    vai_chover: bool
    dia_quente: bool
    dia_frio: bool
    dia_umido: bool
    dia_ventoso: bool
    dia_confortavel: bool
    descricao: str

# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================

def calcular_features(data_alvo):
    """Calcula features para previsão"""
    data_alvo = pd.to_datetime(data_alvo)
    df_hist = df_historico.tail(365).copy()
    
    features = {}
    
    # Features temporais
    features['Ano'] = data_alvo.year
    features['Mes'] = data_alvo.month
    features['Dia'] = data_alvo.day
    features['DiaDoAno'] = data_alvo.dayofyear
    features['Mes_sin'] = np.sin(2 * np.pi * data_alvo.month / 12)
    features['Mes_cos'] = np.cos(2 * np.pi * data_alvo.month / 12)
    features['Dia_sin'] = np.sin(2 * np.pi * data_alvo.day / 31)
    features['Dia_cos'] = np.cos(2 * np.pi * data_alvo.day / 31)
    
    # Lag features
    for var in modelo['variaveis']:
        if var in df_hist.columns:
            features[f'{var}_lag1'] = df_hist[var].iloc[-1]
            features[f'{var}_lag7'] = df_hist[var].iloc[-7]
            features[f'{var}_lag30'] = df_hist[var].iloc[-30]
    
    # Médias móveis
    for var in modelo['variaveis']:
        if var in df_hist.columns:
            features[f'{var}_ma7'] = df_hist[var].tail(7).mean()
            features[f'{var}_ma30'] = df_hist[var].tail(30).mean()
    
    # Variações
    for var in modelo['variaveis']:
        if var in df_hist.columns:
            features[f'{var}_diff1'] = df_hist[var].iloc[-1] - df_hist[var].iloc[-2]
            features[f'{var}_diff7'] = df_hist[var].iloc[-1] - df_hist[var].iloc[-7]
    
    # Estatísticas
    for var in modelo['variaveis']:
        if var in df_hist.columns:
            features[f'{var}_std7'] = df_hist[var].tail(7).std()
            features[f'{var}_max7'] = df_hist[var].tail(7).max()
            features[f'{var}_min7'] = df_hist[var].tail(7).min()
    
    # Features sazonais
    features['Trimestre'] = (data_alvo.month - 1) // 3 + 1
    features['Estacao'] = 1 if data_alvo.month in [12,1,2] else 2 if data_alvo.month in [3,4,5] else 3 if data_alvo.month in [6,7,8] else 4
    features['FimDeSemana'] = 1 if data_alvo.dayofweek in [5,6] else 0
    features['DiaSemana'] = data_alvo.dayofweek
    
    return features

def fazer_previsao(data_str):
    """Faz previsão para uma data"""
    try:
        if '/' in data_str:
            data = pd.to_datetime(data_str, format='%d/%m/%Y')
        else:
            data = pd.to_datetime(data_str)
        
        features_dict = calcular_features(data)
        X = np.array([[features_dict.get(name, 0) for name in modelo['feature_names']]])
        
        previsoes = {}
        for variavel, model in modelo['modelos'].items():
            scaler = modelo['scalers'][variavel]
            X_scaled = scaler.transform(X)
            previsao = model.predict(X_scaled)[0]
            previsoes[variavel] = previsao
        
        return data, previsoes
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro: {str(e)}")

# =====================================================
# ENDPOINTS
# =====================================================

@app.get("/")
def root():
    """Página inicial"""
    return {
        "app": "🌤️ API de Previsão Climática",
        "cidade": modelo['cidade'],
        "status": "online ✅",
        "endpoints": {
            "previsao": "POST /previsao",
            "info": "GET /info",
            "docs": "GET /docs"
        },
        "exemplo_uso": {
            "url": "/previsao",
            "method": "POST",
            "body": {"data": "2025-10-04"}
        }
    }

@app.get("/health")
def health():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "modelo": modelo['cidade']
    }

@app.get("/info")
def info():
    """Informações do modelo"""
    return {
        "cidade": modelo['cidade'],
        "data_treinamento": modelo['data_treinamento'],
        "numero_features": len(modelo['feature_names']),
        "variaveis": modelo['variaveis'],
        "dados_disponiveis": len(df_historico)
    }

@app.post("/previsao", response_model=PrevisaoResponse)
def prever_clima(request: PrevisaoRequest):
    """Previsão climática"""
    
    data, previsoes = fazer_previsao(request.data)
    
    vai_chover = previsoes['Precipitação (mm)'] > 1
    dia_quente = previsoes['Temp Máx (°C)'] > 30
    dia_frio = previsoes['Temp Mín (°C)'] < 15
    dia_umido = previsoes['Umidade (%)'] > 70
    dia_ventoso = previsoes['Vento (km/h)'] > 20
    dia_confortavel = (
        20 <= previsoes['Temp Média (°C)'] <= 28 and
        40 <= previsoes['Umidade (%)'] <= 70 and
        previsoes['Precipitação (mm)'] < 5
    )
    
    descricoes = []
    if vai_chover:
        descricoes.append("☔ Vai chover")
    if dia_quente:
        descricoes.append("🔥 Dia quente")
    if dia_frio:
        descricoes.append("❄️ Dia frio")
    if dia_confortavel:
        descricoes.append("😊 Dia confortável")
    
    descricao = ", ".join(descricoes) if descricoes else "🌤️ Dia normal"
    
    dias_semana = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    
    return PrevisaoResponse(
        data=data.strftime('%d/%m/%Y'),
        cidade=modelo['cidade'],
        dia_semana=dias_semana[data.dayofweek],
        temperatura_media=round(previsoes['Temp Média (°C)'], 1),
        temperatura_maxima=round(previsoes['Temp Máx (°C)'], 1),
        temperatura_minima=round(previsoes['Temp Mín (°C)'], 1),
        umidade=round(previsoes['Umidade (%)'], 1),
        precipitacao=round(previsoes['Precipitação (mm)'], 1),
        vento=round(previsoes['Vento (km/h)'], 1),
        vai_chover=vai_chover,
        dia_quente=dia_quente,
        dia_frio=dia_frio,
        dia_umido=dia_umido,
        dia_ventoso=dia_ventoso,
        dia_confortavel=dia_confortavel,
        descricao=descricao
    )
#teste
# Para testes locais
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
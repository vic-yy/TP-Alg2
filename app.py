from flask import Flask
import dash
from dash import html
import dash_leaflet as dl
import pandas as pd
from sqlalchemy import create_engine
import requests
from time import sleep
import os
from tqdm import tqdm

# === Configurações ===
CSV_PATH = "csv/arquivo.csv"
DB_PATH = "bares.db"

# === Funções ===
def montar_endereco(row):
    partes = [
        row.get("DESC_LOGRADOURO", ""),
        row.get("NOME_LOGRADOURO", ""),
        str(row.get("NUMERO_IMOVEL", "")),
        row.get("COMPLEMENTO", ""),
        row.get("NOME_BAIRRO", ""),
        "Minas Gerais", "Brasil"
    ]
    return " ".join(str(p) for p in partes if pd.notna(p) and p != "")

def obter_coordenadas(endereco):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": endereco, "format": "json", "limit": 1}
    headers = {"User-Agent": "mapa-bares-app"}
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    if data:
        return float(data[0]["lat"]), float(data[0]["lon"])
    return None

# === Prepara dados e banco ===
if not os.path.exists(DB_PATH):
    df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8")
    df = df[df["CNAE_PRINCIPAL"].isin([5611203, 5611204, 5611205])].reset_index(drop=True)
    df["latitude"] = None
    df["longitude"] = None

    for i, row in tqdm(df.iterrows(), total=len(df)):
        endereco = montar_endereco(row)
        coords = obter_coordenadas(endereco)
        if coords:
            df.at[i, "latitude"] = coords[0]
            df.at[i, "longitude"] = coords[1]
        sleep(1)  # para não sobrecarregar a API

    engine = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql("bares", con=engine, index=False, if_exists="replace")
else:
    df = pd.read_sql("SELECT * FROM bares", con=create_engine(f"sqlite:///{DB_PATH}"))

# === Flask + Dash ===
app = Flask(__name__)

dash_app = dash.Dash(__name__, server=app, url_base_pathname="/dash/")
markers = [
    dl.Marker(position=[row["latitude"], row["longitude"]],
              children=[dl.Tooltip(row.get("NOME", "Sem nome"))])
    for _, row in df.dropna(subset=["latitude", "longitude"]).iterrows()
]

dash_app.layout = html.Div([
    html.H2("Mapa dos Bares de BH"),
    dl.Map(center=[-19.9167, -43.9345], zoom=20, children=[
        dl.TileLayer(), *markers
    ], style={'width': '100%', 'height': '600px'})
])

if __name__ == "__main__":
    app.run(debug=True, port=5000)

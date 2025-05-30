import dash
import dash_leaflet as dl
from dash import html, Output, Input, State
import pandas as pd
from scipy.spatial import KDTree

# === Carregar e preparar dados ===
df = pd.read_csv("bares_e_restaurantes_filtrados.csv", sep=";")
df = df.dropna(subset=["latitude", "longitude"]).copy()

# Montar endereço formatado
def montar_endereco(row):
    partes = [
        str(row.get("DESC_LOGRADOURO", "")),
        str(row.get("NOME_LOGRADOURO", "")),
        str(row.get("NUMERO_IMOVEL", "")),
        str(row.get("COMPLEMENTO", "")),
        str(row.get("NOME_BAIRRO", "")),
        "Belo Horizonte - MG"
    ]
    return ", ".join([p for p in partes if p.strip() and p != "nan"])

df["endereco_formatado"] = df.apply(montar_endereco, axis=1)

# Criar K-D Tree
coordenadas = df[["latitude", "longitude"]].values
kd_tree = KDTree(coordenadas)

# === App ===
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Bares e Restaurantes de BH"),
    html.Button("Limpar seleção", id="btn_reset", n_clicks=0),

    dl.Map(
        id="mapa",
        center=[-19.92, -43.94], zoom=13,
        style={'width': '100%', 'height': '500px'},
        children=[
            dl.TileLayer(),
            dl.FeatureGroup([
                dl.EditControl(
                    id="edit_control",
                    draw={"rectangle": True, "polygon": False, "marker": False, "circle": False},
                    edit={"edit": False}
                ),
                dl.LayerGroup(id="layer_markers")
            ])
        ]
    ),

    html.H4("Estabelecimentos Selecionados"),
    html.Div(id="tabela_resultado")
])

# === Callback principal: seleção no mapa ===
@app.callback(
    [Output("tabela_resultado", "children"),
     Output("layer_markers", "children")],
    [Input("edit_control", "geojson"),
     Input("btn_reset", "n_clicks")]
)
def atualizar_resultado(geojson, n_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate()

    trigger = ctx.triggered[0]["prop_id"]

    if "btn_reset" in trigger or not geojson or not geojson.get("features"):
        return html.Div("Selecione uma área no mapa."), []

    try:
        # Extrair coordenadas do retângulo desenhado
        coords = geojson["features"][0]["geometry"]["coordinates"][0]
        lons = [p[0] for p in coords]
        lats = [p[1] for p in coords]
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)

        # Buscar com K-D Tree
        centro = [(lat_min + lat_max)/2, (lon_min + lon_max)/2]
        raio = max(abs(lat_max - lat_min), abs(lon_max - lon_min)) * 0.75
        indices = kd_tree.query_ball_point(centro, raio)

        # Filtrar os que realmente estão dentro do retângulo
        filtrado = df.iloc[indices]
        filtrado = filtrado[
            (filtrado["latitude"] >= lat_min) & (filtrado["latitude"] <= lat_max) &
            (filtrado["longitude"] >= lon_min) & (filtrado["longitude"] <= lon_max)
        ]

        # Montar a tabela
        tabela = html.Table([
            html.Thead(html.Tr([
                html.Th("Nome ou Fantasia"),
                html.Th("Data Início"),
                html.Th("Possui Alvará"),
                html.Th("Endereço")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(row.get("NOME_FANTASIA") or row.get("NOME", "")),
                    html.Td(row.get("DATA_INICIO_ATIVIDADE", "")),
                    html.Td("Sim" if str(row.get("IND_POSSUI_ALVARA", "")).strip().upper() == "S" else "Não"),
                    html.Td(row["endereco_formatado"])
                ]) for _, row in filtrado.iterrows()
            ])
        ], style={"width": "100%", "border": "1px solid black", "borderCollapse": "collapse"})

        # Marcadores no mapa
        marcadores = [
            dl.Marker(position=[row["latitude"], row["longitude"]],
                      children=dl.Tooltip(row.get("NOME_FANTASIA") or row.get("NOME", "Sem nome")))
            for _, row in filtrado.iterrows()
        ]

        return tabela, marcadores

    except Exception as e:
        return html.Div(f"Erro ao processar seleção: {e}"), []

# === Rodar o servidor ===
if __name__ == "__main__":
    app.run_server(debug=True)

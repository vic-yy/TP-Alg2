import pandas as pd
import numpy as np
import dash
from dash import html, dash_table, dcc
from dash import callback_context
import dash_leaflet as dl
from dash.dependencies import Input, Output, State
import re
import os
from pyproj import Transformer
import json

# ---------------------------------------------------------
# 1) CARREGA DADOS DE BARES E FAZ CONVERSÃO UTM → LAT/LON
# ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'df_com_buteco_v2.csv')

df = pd.read_csv(data_path, encoding='utf-8', low_memory=False, sep=',')

transformer = Transformer.from_crs("EPSG:31983", "EPSG:4326", always_xy=True)
def utm_to_latlon(geometry):
    try:
        x, y = map(float, re.findall(r'POINT \(([\d.]+) ([\d.]+)\)', geometry)[0])
        lon, lat = transformer.transform(x, y)
        return lat, lon
    except:
        return np.nan, np.nan

df[['latitude', 'longitude']] = df['GEOMETRIA'].apply(utm_to_latlon).apply(pd.Series)
df = df.dropna(subset=['latitude', 'longitude'])

df['NOME_FANTASIA'] = df['NOME_FANTASIA'].fillna(df['NOME'])
df['FULL_ADDRESS'] = df.apply(
    lambda row: f"{row['DESC_LOGRADOURO']} {row['NOME_LOGRADOURO']}, {row['NUMERO_IMOVEL']}"
                f"{', ' + row['COMPLEMENTO'] if pd.notna(row['COMPLEMENTO']) else ''}, {row['NOME_BAIRRO']}",
    axis=1
)

df['DATA_INICIO_ATIVIDADE'] = pd.to_datetime(df['DATA_INICIO_ATIVIDADE'], format='%d-%m-%Y', errors='coerce')
df['DATA_INICIO_STR'] = df['DATA_INICIO_ATIVIDADE'].dt.strftime('%Y-%m-%d')

df = df[['NOME_FANTASIA', 'latitude', 'longitude', 'FULL_ADDRESS',
         'DATA_INICIO_ATIVIDADE', 'IND_POSSUI_ALVARA', 'DATA_INICIO_STR',
         'Comida de Boteco', 'NOME_BAIRRO']]

# ---------------------------------------------------------
# 2) MONTA K-D TREE PARA BUSCAR PONTOS NO RETÂNGULO VISÍVEL
# ---------------------------------------------------------
class KDTreeNode:
    def __init__(self, point, data, axis, left=None, right=None):
        self.point = point
        self.data = data
        self.axis = axis
        self.left = left
        self.right = right

def build_kd_tree(points, data, depth=0):
    if not points:
        return None
    axis = depth % 2
    sorted_indices = np.argsort([p[axis] for p in points])
    points = [points[i] for i in sorted_indices]
    data = [data[i] for i in sorted_indices]
    mid = len(points) // 2
    return KDTreeNode(
        point=points[mid],
        data=data[mid],
        axis=axis,
        left=build_kd_tree(points[:mid], data[:mid], depth + 1),
        right=build_kd_tree(points[mid + 1:], data[mid + 1:], depth + 1)
    )

def range_search(node, rect, result, depth=0):
    if node is None:
        return
    axis = depth % 2
    min_val, max_val = rect[axis], rect[axis + 2]
    point_val = node.point[axis]

    if min_val <= point_val <= max_val:
        lon, lat = node.point
        if rect[0] <= lon <= rect[2] and rect[1] <= lat <= rect[3]:
            result.append(node.data)

    if point_val >= min_val:
        range_search(node.left, rect, result, depth + 1)
    if point_val <= max_val:
        range_search(node.right, rect, result, depth + 1)

points = df[['longitude', 'latitude']].values.tolist()
data = df.index.tolist()
kd_tree = build_kd_tree(points, data)

# ---------------------------------------------------------
# 3) CARREGA E REPROJETA O GEOJSON DOS BAIRROS (EPSG:32723 → 4326)
# ---------------------------------------------------------
bairros_geojson_path = os.path.join(script_dir, 'bairro_popular_lei1069814012014.geojson')
with open(bairros_geojson_path, encoding='utf-8') as f:
    bairros_raw = json.load(f)

# Transformer de EPSG:32723 → EPSG:4326
transformer_bairros = Transformer.from_crs("EPSG:32723", "EPSG:4326", always_xy=True)

def reproject_coords_to_lonlat(xy):
    # Recebe [x, y] em EPSG:32723, devolve [lon, lat] em 4326
    lon, lat = transformer_bairros.transform(xy[0], xy[1])
    return [lon, lat]

def reproject_geometry(geometry):
    # Trabalha apenas com MultiPolygon conforme seu GeoJSON
    geom_type = geometry['type']
    if geom_type == 'MultiPolygon':
        new_coords = []
        for polygon in geometry['coordinates']:
            new_polygon = []
            for ring in polygon:
                # cada ring é lista de pontos [[x,y], [x,y], ...]
                new_ring = [reproject_coords_to_lonlat(pt) for pt in ring]
                new_polygon.append(new_ring)
            new_coords.append(new_polygon)
        return {'type': 'MultiPolygon', 'coordinates': new_coords}
    # se por acaso houver outro tipo (não esperado), devolve original
    return geometry

# Monta o GeoJSON reprojetado para WGS84
bairros_geojson = {"type": bairros_raw["type"], "features": []}
for feat in bairros_raw["features"]:
    new_feat = {
        "type": feat["type"],
        "properties": feat["properties"],
        "geometry": reproject_geometry(feat["geometry"])
    }
    bairros_geojson["features"].append(new_feat)

# ---------------------------------------------------------
# 4) CONFIGURA APP DASH + LAYERS
# ---------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = 'BH - Bares e Hestaurantes'

# Necessário para hover/click em markers no dash_leaflet
app.scripts.append_script({
    'external_url': 'https://unpkg.com/leaflet@1.7.1/dist/leaflet.js'
})

            
filter_style = {
    'marginBottom': '0px',
    'padding': '0px 5px',  # padding lateral apenas, evita "empurrar" verticalmente
    'width': '90px',
    'font-weight': 'bold'
}

app.layout = html.Div([
    dcc.Store(id='initial-bounds', data=[[-20, -44], [-19, -43]]),
    dcc.Interval(id='startup-trigger', interval=100, n_intervals=0, max_intervals=1),
    dcc.Store(id='geojson-store'),
    dcc.Store(id='cluster-enabled', data=False),
    dcc.Store(id='comida-filter-state', data=False),  # False = Todos

    # ───────────────────────────────────────────────────────────
    # LEFT: MAPA
    # ───────────────────────────────────────────────────────────
    html.Div([
        dl.Map(
            id='map',
            style={'width': '100%', 'height': '100%'},
            center=[-19.8800, -43.9378],
            zoom=12,
            children=[
                dl.TileLayer(),

                # camada vazia para bairros; será preenchida pelo callback
                dl.LayerGroup(id='bairros-group'),

                dl.FeatureGroup([
                    dl.EditControl(
                        id="edit_control",
                        draw={"rectangle": True, "polygon": False, "marker": False,
                              "circle": False, "polyline": False, "circlemarker": False},
                        edit={"edit": False, "remove": True}
                    ),
                    dl.LayerGroup(id="markers")
                ])
            ]
        )
    ], style={
        'flex': '1',
        'minWidth': '50%',
        'padding': '15px',
        'boxSizing': 'border-box',
        'height': '100%'
    }),

    # ───────────────────────────────────────────────────────────
    # RIGHT: TABELA + FILTROS
    # ───────────────────────────────────────────────────────────
    html.Div([
        html.Div("Bares e Restaurantes de BH", style={
            'fontSize': '30px',
            'fontWeight': 'bold',
            'color': '#f2464d',
            'textAlign': 'center',
            'paddingBottom': '10px',
            'paddingTop': '0px',
            'marginBottom': '20px',
            'borderBottom': '5px solid #f2464d'
        }),
        html.Div([
            # Coluna Esquerda: Botão Reset e Toggle Bairros
            html.Div([
                # Botão Bairros
                html.Div([
                    html.Button(
                        'Bairros',
                        id='cluster-toggle-btn',
                        n_clicks=0,
                        style={
                            'height': '32px',
                            'backgroundColor': 'white',
                            'color': "#12A824",
                            'border': '2px solid #12A824',
                            'borderRadius': '4px',
                            'cursor': 'pointer',
                            'width': '180px',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'padding': '0 12px',
                            'fontSize': '14px',
                            'marginBottom': '10px'
                        }
                    )
                ]),

                # Botão Comida di Buteco
                html.Div([
                    html.Button(
                        'Comida di Buteco',
                        id='comida-toggle-btn',
                        n_clicks=0,
                        style={
                            'height': '32px',
                            'backgroundColor': 'white',
                            'color': "#f2464d",
                            'border': '2px solid #f2464d',
                            'borderRadius': '4px',
                            'cursor': 'pointer',
                            'width': '180px',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'padding': '0 12px',
                            'fontSize': '14px',
                            'marginBottom': '10px'
                        }
                    ),
                ]),

                # Botão Contorno
                html.Div([
                    html.Button(
                        'Contorno',
                        id='bairros-toggle-btn',
                        n_clicks=0,
                        style={
                            'height': '32px',
                            'backgroundColor': 'white',
                            'color': '#111111',
                            'border': '2px solid #111111',
                            'borderRadius': '4px',
                            'cursor': 'pointer',
                            'width': '180px',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'padding': '0 12px',
                            'fontSize': '14px'
                        }
                    )
                ]),
            ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center'}),

            # Coluna Direita: Filtros de Nome e Endereço
            html.Div([
                # Filtro por Nome
                html.Div([
                    html.Label('Nome:', style=filter_style),
                    dcc.Input(
                        id='name-filter',
                        type='text',
                        placeholder='Buscar por nome...',
                        debounce=True,
                        style={'width': '100%', 'height': '32px'}
                    )
                ], style={
                    'display': 'flex',
                    'flexDirection': 'row',
                    'alignItems': 'center',  # ← esta linha alinha verticalmente os filhos
                    'marginBottom': '10px',
                    'marginRight': '20px'  # Adiciona margem direita para espaçamento
                }),
                # Filtro por Endereço
                html.Div([
                    html.Label('Endereço:', style=filter_style),
                    dcc.Input(
                        id='address-filter',
                        type='text',
                        placeholder='Buscar por endereço...',
                        debounce=True,
                        style={'width': '100%', 'height': '32px'}
                    )
                ], style={
                    'display': 'flex',
                    'flexDirection': 'row',
                    'alignItems': 'center',  # ← esta linha alinha verticalmente os filhos
                    'marginBottom': '10px',
                    'marginRight': '20px'  # Adiciona margem direita para espaçamento
                }),
                # Filtro Alvará
                html.Div([
                    html.Label('Alvará:', style=filter_style),
                    dcc.Checklist(
                        id='alvara-filter',
                        options=[{'label': 'Sim', 'value': 'SIM'},
                                {'label': 'Não', 'value': 'NÃO'}],
                        value=['SIM', 'NÃO'],
                        labelStyle={'display': 'inline-block', 'marginRight': '10px'},
                        style={
                            'display': 'flex',
                            'flexDirection': 'row',
                            'alignItems': 'center',
                            'padding': '0',
                            'marginLeft': '-5px'
                        }
                    )
                ], style={
                    'display': 'flex',
                    'flexDirection': 'row',
                    'marginTop': '15px',
                    'alignItems': 'center',  # ← esta linha alinha verticalmente os filhos
                    'marginBottom': '0px'
                }),
            ], style={'flex': '3'}),
        ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '10px'}),
        
        # Tabela rolável
        html.Div([
            dash_table.DataTable(
                id='table',
                columns=[
                    {'name': 'Nome', 'id': 'NOME_FANTASIA'},
                    {'name': 'Endereço', 'id': 'FULL_ADDRESS'},
                    {'name': 'Início', 'id': 'DATA_INICIO_STR'},
                    {'name': 'Alvará', 'id': 'IND_POSSUI_ALVARA'}
                ],
                page_size=10,
                sort_action='native',
                style_table={'height': '100%', 'overflowY': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'padding': '8px'
                },
                style_data_conditional=[
                    {'if': {'column_id': 'DATA_INICIO_STR'},
                    'width': '100px', 'minWidth': '100px', 'maxWidth': '100px',
                    'textAlign': 'center'},
                    {'if': {'column_id': 'IND_POSSUI_ALVARA'}, 'textAlign': 'center'},
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#f0f0f0'},
                    {'if': {'state': 'selected'},
                    'backgroundColor': '#d3e0ea',
                    'border': '1px solid #0074D9'}
                ],
                style_header={'textAlign': 'center', 'paddingLeft': '6px'},
            )
        ], style={'flex': '1', 'overflow': 'auto', 'marginTop': '10px'})
    ], style={
        'flex': '1',
        'minWidth': '300px',
        'height': '100%',
        'display': 'flex',
        'flexDirection': 'column',
        'padding': '15px',
        'boxSizing': 'border-box'
    })
], style={
    'margin': '-10px',
    'padding': '0',
    'height': '100vh',
    'width': '100vw',
    'overflow': 'hidden',
    'display': 'flex',
    'flexDirection': 'row',
    'backgroundColor': "#ffe334"
})

# ───────────────────────────────────────────────────────────
# CALLBACK: Atualiza marcadores conforme retângulo visível
# ───────────────────────────────────────────────────────────
@app.callback(
    Output('markers', 'children'),
    [
        Input('map', 'bounds'),
        Input('initial-bounds', 'data'),
        Input('table', 'selected_rows'),
        Input('edit_control', 'geojson'),
        Input('alvara-filter', 'value'),
        Input('name-filter', 'value'),
        Input('address-filter', 'value'),
        Input('comida-filter-state', 'data'),
        Input('cluster-enabled', 'data')  
    ],
    [
        State('table', 'data'),
        State('markers', 'children')
    ]
)
def update_markers(bounds, initial_bounds, selected_rows, geojson,
                   alvara_filter, name_filter, address_filter,
                   comida_boteco_filter, cluster_enabled,
                   table_data, current_markers):
    ctx = callback_context
    triggered_prop = ctx.triggered[0]['prop_id'] if ctx.triggered else ''

    result = set()

    # Verifica se há retângulos desenhados
    has_valid_rectangle = False
    if geojson and geojson.get("features"):
        for feature in geojson["features"]:
            geometry = feature.get("geometry")
            if geometry and geometry.get("type") == "Polygon":
                coords = geometry["coordinates"][0]
                if coords:
                    has_valid_rectangle = True
                    lons = [p[0] for p in coords]
                    lats = [p[1] for p in coords]
                    lat_min, lat_max = min(lats), max(lats)
                    lon_min, lon_max = min(lons), max(lons)
                    rect = [lon_min, lat_min, lon_max, lat_max]
                    partial_result = []
                    range_search(kd_tree, rect, partial_result)
                    result.update(partial_result)

    # Se não houver retângulos válidos, usa os bounds do mapa como fallback
    if not has_valid_rectangle:
        bounds = bounds or initial_bounds
        if not bounds:
            return current_markers or []
        lat_min, lon_min = bounds[0]
        lat_max, lon_max = bounds[1]
        rect = [lon_min, lat_min, lon_max, lat_max]
        partial_result = []
        range_search(kd_tree, rect, partial_result)
        result.update(partial_result)

    # Converte para lista de índices
    filtered_df = df.loc[list(result)]

    name_filter = name_filter if isinstance(name_filter, str) else ''
    address_filter = address_filter if isinstance(address_filter, str) else ''

    if comida_boteco_filter:  # se True, mostra apenas participantes
        filtered_df = filtered_df[filtered_df['Comida de Boteco'] == 1]
    if alvara_filter:
        filtered_df = filtered_df[filtered_df['IND_POSSUI_ALVARA'].str.upper().isin(alvara_filter)]
    if name_filter:
        filtered_df = filtered_df[filtered_df['NOME_FANTASIA'].str.contains(name_filter, case=False, na=False)]
    if address_filter:
        filtered_df = filtered_df[filtered_df['FULL_ADDRESS'].str.contains(address_filter, case=False, na=False)]

    # Limita total de marcadores no mapa
    MAX_TOTAL_MARKERS = 200

    # Separação dos participantes e outros
    df_boteco = filtered_df[filtered_df['Comida de Boteco'] == 1]
    df_outros = filtered_df[filtered_df['Comida de Boteco'] == 0]

    # Contagem máxima respeitando o total de 300
    n_participantes = len(df_boteco)
    remaining_slots = MAX_TOTAL_MARKERS - n_participantes

    # Amostragem aleatória dos participantes, se exceder
    if n_participantes > MAX_TOTAL_MARKERS:
        df_boteco = df_boteco.sample(n=MAX_TOTAL_MARKERS, random_state=42)
        df_outros = df_outros.iloc[0:0]
    else:
        # Amostragem estratificada dos não participantes por bairro
        if len(df_outros) > remaining_slots:
            grupos = df_outros.groupby('NOME_BAIRRO')
            n_bairros = grupos.ngroups
            por_bairro = max(1, remaining_slots // max(n_bairros, 1))  # Evita divisão por zero

            df_outros_amostrados = []
            for _, grupo in grupos:
                df_outros_amostrados.append(grupo.sample(n=min(por_bairro, len(grupo)), random_state=42))

            df_outros = pd.concat(df_outros_amostrados)

    selected_name = None
    if selected_rows and table_data:
        selected_name = table_data[selected_rows[0]]['NOME_FANTASIA']

    markers = []

    # Participantes (sempre mostrados individualmente)
    for _, row in df_boteco.iterrows():
        icon_url = 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png'
        icon = {
            'iconUrl': icon_url,
            'iconSize': [25, 41],
            'iconAnchor': [12, 41],
            'popupAnchor': [1, -34],
            'shadowSize': [41, 41]
        }
        markers.append(
            dl.Marker(
                position=[row['latitude'], row['longitude']],
                icon=icon,
                children=[
                    dl.Popup(html.Div([
                        html.H4(row['NOME_FANTASIA'], style={'fontSize': '16px', 'margin': '0'}),
                        html.P(row['FULL_ADDRESS'], style={'fontSize': '14px', 'margin': '5px 0'})
                    ])),
                    dl.Tooltip(html.Div([
                        html.H4(row['NOME_FANTASIA'], style={'fontSize': '16px', 'margin': '0'}),
                        html.P(row['FULL_ADDRESS'], style={'fontSize': '14px', 'margin': '5px 0'})
                    ]))
                ],
                id={'type': 'marker', 'index': row['NOME_FANTASIA']}
            )
        )
    n_participantes = len(df_boteco)
    remaining_slots = MAX_TOTAL_MARKERS - n_participantes

    if remaining_slots <= 0:
        df_boteco = df_boteco.head(MAX_TOTAL_MARKERS)
        df_outros = df_outros.iloc[0:0]
    else:
        df_outros = df_outros.head(remaining_slots)

    if cluster_enabled:
        # Agrupar não participantes por bairro
        grouped = df_outros.groupby('NOME_BAIRRO')
        total_por_bairro = grouped.size().to_dict()
        for bairro, group in grouped:
            lat = group['latitude'].mean()
            lon = group['longitude'].mean()
            total = total_por_bairro.get(bairro, len(group))
            markers.append(
                dl.Marker(
                    position=[lat, lon],
                    icon={
                        'iconUrl': 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
                        'iconSize': [25, 40],
                        'iconAnchor': [17, 55],
                        'popupAnchor': [1, -34],
                        'shadowSize': [41, 41]
                    },
                    children=[
                        dl.Tooltip(html.Div([
                            html.H4(f"{bairro}", style={'fontSize': '16px', 'margin': '0'}),
                            html.P(f"{total} BARES", style={'fontSize': '14px', 'margin': '5px 0'})
                        ]))
                    ],
                    id={'type': 'cluster', 'index': bairro}
                )
            )
    else:
        for _, row in df_outros.iterrows():
            icon_url = 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png'
            icon = {
                'iconUrl': icon_url,
                'iconSize': [25, 40],
                'iconAnchor': [12, 41],
                'popupAnchor': [1, -34],
                'shadowSize': [41, 41]
            }
            markers.append(
                dl.Marker(
                    position=[row['latitude'], row['longitude']],
                    icon=icon,
                    children=[
                        dl.Popup(html.Div([
                            html.H4(row['NOME_FANTASIA'], style={'fontSize': '16px', 'margin': '0'}),
                            html.P(row['FULL_ADDRESS'], style={'fontSize': '14px', 'margin': '5px 0'})
                        ]))
                    ],
                    id={'type': 'marker', 'index': row['NOME_FANTASIA']}
                )
            )

    return markers

# ───────────────────────────────────────────────────────────
# CALLBACK: Exibe ou oculta a camada de bairros
# ───────────────────────────────────────────────────────────
@app.callback(
    Output('bairros-group', 'children'),
    Output('bairros-toggle-btn', 'children'),
    Output('bairros-toggle-btn', 'style'),
    Input('bairros-toggle-btn', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_bairros_layer(n_clicks):
    is_on = n_clicks % 2 == 1
    label = "Contorno"

    button_style_active = {
        'backgroundColor': '#111111',
        'color': 'white',
        'border': 'none',
        'borderRadius': '4px',
        'cursor': 'pointer',
        'width': '180px',
        'height': '30px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'padding': '0 12px',
        'fontSize': '14px'
    }

    button_style_inactive = {
        'backgroundColor': 'white',
        'color': '#111111',
        'border': '2px solid #111111',
        'borderRadius': '4px',
        'cursor': 'pointer',
        'width': '180px',
        'height': '30px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'padding': '0 12px',
        'fontSize': '14px'
    }

    if is_on:
        return [
            dl.GeoJSON(
                data=bairros_geojson,
                options=dict(style=dict(
                    color='black',
                    weight=3,
                    dashArray='10,10',
                    fillOpacity=0
                ))
            )
        ], label, button_style_active
    else:
        return [], label, button_style_inactive

# ───────────────────────────────────────────────────────────
# CALLBACK: Sincronia do GeoJSON desenhado (não mexer)
# ───────────────────────────────────────────────────────────
@app.callback(
    Output('geojson-store', 'data'),
    Input('edit_control', 'geojson')
)
def sync_geojson(geojson):
    return geojson

# ───────────────────────────────────────────────────────────
# CALLBACK: Alterna estado de clusterização
# ───────────────────────────────────────────────────────────
@app.callback(
    Output('cluster-enabled', 'data'),
    Output('cluster-toggle-btn', 'children'),
    Output('cluster-toggle-btn', 'style'),
    Output('comida-filter-state', 'data'),
    Output('comida-toggle-btn', 'children'),
    Output('comida-toggle-btn', 'style'),
    Input('cluster-toggle-btn', 'n_clicks'),
    Input('comida-toggle-btn', 'n_clicks'),
    State('cluster-enabled', 'data'),
    State('comida-filter-state', 'data'),
    prevent_initial_call=True
)
def handle_cluster_and_comida_toggle(n_cluster_clicks, n_comida_clicks, cluster_state, comida_state):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Estilos
    cluster_style_active = {
        'backgroundColor': '#12A824', 'color': 'white', 'border': 'none', 'borderRadius': '4px',
        'cursor': 'pointer', 'width': '180px', 'height': '32px', 'display': 'flex', 'marginBottom': '10px',
        'alignItems': 'center', 'justifyContent': 'center', 'padding': '0 12px', 'fontSize': '14px'
    }
    cluster_style_inactive = {
        'backgroundColor': 'white', 'color': '#12A824', 'border': '2px solid #12A824', 'borderRadius': '4px',
        'cursor': 'pointer', 'width': '180px', 'height': '32px', 'display': 'flex', 'marginBottom': '10px',
        'alignItems': 'center', 'justifyContent': 'center', 'padding': '0 12px', 'fontSize': '14px'
    }

    comida_style_active = {
        'backgroundColor': '#f2464d', 'color': 'white', 'border': 'none', 'borderRadius': '4px',
        'cursor': 'pointer', 'width': '180px', 'height': '32px', 'display': 'flex', 'marginBottom': '10px',
        'alignItems': 'center', 'justifyContent': 'center', 'padding': '0 12px', 'fontSize': '14px'
    }
    comida_style_inactive = {
        'backgroundColor': 'white', 'color': '#f2464d', 'border': '2px solid #f2464d', 'borderRadius': '4px',
        'cursor': 'pointer', 'width': '180px', 'height': '32px', 'display': 'flex', 'marginBottom': '10px',
        'alignItems': 'center', 'justifyContent': 'center', 'padding': '0 8px', 'fontSize': '14px', 'lineHeight': '1'
    }

    # Se foi o botão "Comida di Buteco"
    if triggered_id == 'comida-toggle-btn':
        new_comida_state = not comida_state if comida_state is not None else True
        new_cluster_state = False  # sempre desativa cluster ao ativar "comida"
        return (
            new_cluster_state,
            "Bairros",  # botão de bairros mantém mesmo label
            cluster_style_inactive,
            new_comida_state,
            "Comida di Buteco",
            comida_style_active if new_comida_state else comida_style_inactive
        )

    # Se foi o botão "Bairros"
    elif triggered_id == 'cluster-toggle-btn':
        new_cluster_state = not cluster_state if cluster_state is not None else True
        return (
            new_cluster_state,
            "Bairros",
            cluster_style_active if new_cluster_state else cluster_style_inactive,
            comida_state,  # não mexe no botão comida
            "Comida di Buteco",
            comida_style_active if comida_state else comida_style_inactive
        )

    # Qualquer outra situação
    raise dash.exceptions.PreventUpdate

# ───────────────────────────────────────────────────────────
# CALLBACK: Atualiza tabela conforme filtros e seleção
# ───────────────────────────────────────────────────────────
@app.callback(
    [Output('table', 'data'), Output('table', 'selected_rows')],
    [
        Input('map', 'bounds'),
        Input('initial-bounds', 'data'),
        Input('geojson-store', 'data'), 
        Input('alvara-filter', 'value'),
        Input('name-filter', 'value'),
        Input('address-filter', 'value'),
        Input('comida-filter-state', 'data'),
        Input({'type': 'marker', 'index': dash.dependencies.ALL}, 'n_clicks')
    ],
    [State('table', 'data')]
)
def update_table_and_selection(map_bounds, initial_bounds, geojson,
                               alvara_filter, name_filter, address_filter, 
                               comida_boteco_filter, marker_clicks, table_data):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    # Situação de inicialização
    if not ctx.triggered or 'startup-trigger' in triggered_id:
        return df.to_dict('records'), []

    # Clique em marcador → seleciona linha correspondente
    if isinstance(ctx.triggered[0]["value"], dict) and ctx.triggered[0]["value"] is not None:
        marker_name = ctx.triggered[0]["value"].get("index")
        if table_data:
            for idx, row in enumerate(table_data):
                if row['NOME_FANTASIA'] == marker_name:
                    return dash.no_update, [idx]
        return dash.no_update, []

    # Agora aplicamos filtros (Alvará, nome, endereço)
    filtered_df = df.copy()
    name_filter = name_filter if isinstance(name_filter, str) else ''
    address_filter = address_filter if isinstance(address_filter, str) else ''

    if comida_boteco_filter:  # se True, mostra apenas participantes
        filtered_df = filtered_df[filtered_df['Comida de Boteco'] == 1]
    if alvara_filter:
        filtered_df = filtered_df[filtered_df['IND_POSSUI_ALVARA'].str.upper().isin(alvara_filter)]
    if name_filter:
        filtered_df = filtered_df[filtered_df['NOME_FANTASIA'].str.contains(name_filter, case=False, na=False)]
    if address_filter:
        filtered_df = filtered_df[filtered_df['FULL_ADDRESS'].str.contains(address_filter, case=False, na=False)]

    # Filtragem espacial por retângulo desenhado ou por limites do mapa
    result = set()
    has_valid_rectangle = False

    if geojson and geojson.get("features"):
        for feature in geojson["features"]:
            geometry = feature.get("geometry")
            if geometry and geometry.get("type") == "Polygon":
                coords = geometry["coordinates"][0]
                if not coords:
                    continue
                has_valid_rectangle = True
                lons = [p[0] for p in coords]
                lats = [p[1] for p in coords]
                lat_min, lat_max = min(lats), max(lats)
                lon_min, lon_max = min(lons), max(lons)
                rect = [lon_min, lat_min, lon_max, lat_max]
                partial_result = []
                range_search(kd_tree, rect, partial_result)
                result.update(partial_result)

    if not has_valid_rectangle:
        bounds = map_bounds or initial_bounds
        if bounds:
            lat_min, lon_min = bounds[0]
            lat_max, lon_max = bounds[1]
            rect = [lon_min, lat_min, lon_max, lat_max]
            partial_result = []
            range_search(kd_tree, rect, partial_result)
            result.update(partial_result)

    # Aplica interseção com o dataframe já filtrado por nome, endereço, alvará, etc.
    valid_indices = filtered_df.index.intersection(list(result))
    filtered_df = filtered_df.loc[valid_indices]

    return filtered_df.to_dict('records'), []


# ───────────────────────────────────────────────────────────
# EXECUTA O SERVIDOR
# ───────────────────────────────────────────────────────────
if __name__ == '__main__':

    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8050)))
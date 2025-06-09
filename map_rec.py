import pandas as pd
import numpy as np
import dash
from dash import html, dash_table, dcc, callback_context 
import dash_leaflet as dl
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import re
import os
from pyproj import Transformer
import json

# ---------------------------------------------------------
# 1) CARREGA DADOS DE BARES E FAZ CONVERSÃO UTM → LAT/LON
# ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'df_com_buteco.csv')

df = pd.read_csv(data_path, encoding='utf-8', low_memory=False, sep=',')
df['Comida de Boteco'] = pd.to_numeric(df['Comida de Boteco'], errors='coerce').fillna(0).astype(int)


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
         'DATA_INICIO_ATIVIDADE', 'IND_POSSUI_ALVARA', 'DATA_INICIO_STR', 'Comida de Boteco']]

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
        if rect[0] <= node.point[0] <= rect[2] and rect[1] <= node.point[1] <= rect[3]:
            result.append(node.data)
    
    if point_val >= min_val:
        range_search(node.left, rect, result, depth + 1)
    if point_val <= max_val:
        range_search(node.right, rect, result, depth + 1)

points = df[['latitude', 'longitude']].values.tolist()
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
                new_ring = [reproject_coords_to_lonlat(pt) for pt in ring]
                new_polygon.append(new_ring)
            new_coords.append(new_polygon)
        return {'type': 'MultiPolygon', 'coordinates': new_coords}
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
app = dash.Dash(
    __name__,
    external_stylesheets=[
        'https://codepen.io/chriddyp/pen/bWLwgP.css'
    ],
    external_scripts=[
        'https://unpkg.com/leaflet@1.7.1/dist/leaflet.js'
    ]
)

app.layout = html.Div([
    dcc.Store(id='initial-bounds', data=[[-20, -44], [-19, -43]]),
    dcc.Interval(id='startup-trigger', interval=100, n_intervals=0, max_intervals=1),
    dcc.Store(id='geojson-store'),

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
                        edit={"edit": False,
                              "remove": False
                              },
                        
                    ),
                    dl.LayerGroup(id="markers")
                ])
            ]
        )
    ], style={
        'flex': '1',
        'minWidth': '50%',
        'padding': '10px',
        'boxSizing': 'border-box',
        'height': '100%'
    }),

    # ───────────────────────────────────────────────────────────
    # RIGHT: TABELA + FILTROS
    # ───────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            # Botão Reset
            html.Div([
                html.Button('Reset Filter', id='reset-btn', style={
                    'backgroundColor': '#0074D9',
                    'color': 'white',
                    'border': 'none',
                    'padding': '6px 12px',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'width': 'fit-content',
                    'display': 'flex',
                    'justifyContent': 'center',
                    'alignItems': 'center',
                    'height': '36px'
                })
            ], style={'marginBottom': '10px'}),

            # Filtro por Nome
            html.Div([
                html.Label('Nome:', style={'marginBottom': '0px', 'padding': '5px', 'width': '90px', 'font-weight': 'bold'}),
                dcc.Input(
                    id='name-filter',
                    type='text',
                    placeholder='Buscar por nome...',
                    debounce=True,
                    style={'width': '100%', 'height': '32px'}
                )
            ], style={'marginBottom': '5px', 'display': 'flex', 'flexDirection': 'row'}),

            # Filtro por Endereço
            html.Div([
                html.Label('Endereço:', style={'marginBottom': '0px', 'padding': '5px', 'width': '90px', 'font-weight': 'bold'}),
                dcc.Input(
                    id='address-filter',
                    type='text',
                    placeholder='Buscar por endereço...',
                    debounce=True,
                    style={'width': '100%', 'height': '32px'}
                )
            ], style={'marginBottom': '5px', 'display': 'flex', 'flexDirection': 'row'}),

            # Filtro Alvará
            html.Div([
                html.Label('Alvará:', style={'marginBottom': '0px', 'padding': '5px', 'width': '90px', 'font-weight': 'bold'}),
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
            ], style={'marginBottom': '5px', 'display': 'flex', 'flexDirection': 'row'}),
            
            # Filtro Comida de Boteco
            html.Div([
                html.Label('Comida de Boteco:', style={
                    'marginBottom': '0px',
                    'padding': '5px',
                    'width': '120px',
                    'font-weight': 'bold'
                }),
                dcc.Checklist(
                    id='comida-boteco-filter',
                        options=[
                            {'label': 'Sim', 'value': 1},
                        ],
                    value=[], 
                    labelStyle={'display': 'inline-block', 'marginRight': '10px'},
                    style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'marginLeft': '-5px'
                    }
                )
            ], style={'marginBottom': '5px', 'display': 'flex', 'flexDirection': 'row'}),
            
            # ─── NOVO: Toggle Bairros ───────────────────────────
            html.Div([
                html.Label('Bairros:', style={'marginBottom': '0px', 'padding': '5px', 'width': '90px', 'font-weight': 'bold'}),
                dcc.Checklist(
                    id='bairros-toggle',
                    options=[{'label': '', 'value': 'SHOW'}],
                    value=[],
                    labelStyle={'display': 'inline-block', 'marginRight': '10px'},
                    style={
                        'display': 'flex',
                        'flexDirection': 'row',
                        'alignItems': 'center',
                        'padding': '0',
                        'marginLeft': '-5px'
                    }
                )
            ], style={'marginBottom': '10px', 'display': 'flex', 'flexDirection': 'row'}),
            # ────────────────────────────────────────────────────

        ]),

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
        ], style={'flex': '1', 'overflow': 'auto'})
    ], style={
        'flex': '1',
        'minWidth': '300px',
        'height': '100%',
        'display': 'flex',
        'flexDirection': 'column',
        'padding': '10px',
        'boxSizing': 'border-box'
    })
], style={
    'display': 'flex',
    'flexDirection': 'row',
    'flexWrap': 'wrap',
    'gap': '0px',
    'height': '98vh',
    'width': '98vw',
    'padding': '0px',
    'boxSizing': 'border-box',
    'overflow': 'hidden',
    'backgroundColor': "#ffffff"
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
        Input('comida-boteco-filter', 'value'),
        Input('edit_control', 'geojson'),
        Input('map', 'clickData'),         
    ],
    State('table', 'data'),
)
def update_markers(bounds,
                   initial_bounds,
                   selected_rows,
                   comida_boteco_filter,
                   geojson,                 
                   click_data,              
                   table_data):
    has_rectangle = geojson and geojson.get("features") and any(
        f.get("geometry") and f["geometry"].get("coordinates")
        for f in geojson["features"]
    )

    if has_rectangle:
        feature = next(
            f for f in reversed(geojson["features"])
            if f.get("geometry") and f["geometry"].get("coordinates")
)
        # o EditControl devolve Polygon com coords[0] = lista de pontos
        coords = feature["geometry"]["coordinates"][0]
        lons = [p[0] for p in coords]
        lats = [p[1] for p in coords]
        rect = [min(lats), min(lons), max(lats), max(lons)]
    else:
        bounds = bounds or initial_bounds
        if not bounds:
            return []
        lat_min, lon_min = bounds[0]
        lat_max, lon_max = bounds[1]
        rect = [lat_min, lon_min, lat_max, lon_max]
    
    result = []
    range_search(kd_tree, rect, result)
    filtered_df = df.loc[result]
    if comida_boteco_filter == [1]:
        filtered_df = filtered_df[filtered_df['Comida de Boteco'] == 1]
    # Se houver mais de 200 pontos e nenhum selecionado, amostra aleatoriamente 200
    if not selected_rows or not table_data:
        if len(filtered_df) > 200:
            filtered_df = filtered_df.sample(n=200, random_state=None)
    
    selected_name = None
    if selected_rows and table_data:
        selected_name = table_data[selected_rows[0]]['NOME_FANTASIA']
    
    markers = []
    for _, row in filtered_df.iterrows():
        is_selected = (row['NOME_FANTASIA'] == selected_name)
        if row['Comida de Boteco'] == 1:
         icon_url = 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png'
        else:
         icon_url = 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png'
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
                    dl.Tooltip(html.Div([
                        html.H4(row['NOME_FANTASIA'], style={'fontSize': '16px', 'margin': '0'})
                    ])),
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
    [Input('bairros-toggle', 'value')]
)
def toggle_bairros_layer(selected):
    if 'SHOW' in selected:
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
        ]
    # Se desmarcado ou vazio, não retorna nada
    return []

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
# CALLBACK: Atualiza tabela conforme filtros e seleção
# ───────────────────────────────────────────────────────────
@app.callback(
    [Output('table', 'data'),
     Output('table', 'selected_rows')],
    [
        Input('map', 'bounds'),
        Input('initial-bounds', 'data'),
        Input('edit_control', 'geojson'),
        Input('map', 'clickData'),               
        Input('reset-btn', 'n_clicks'),
        Input('alvara-filter', 'value'),
        Input('name-filter', 'value'),
        Input('address-filter', 'value'),
        Input('comida-boteco-filter', 'value'),
        Input({'type': 'marker', 'index': dash.ALL}, 'n_clicks')
    ],
    State('table', 'data'),
)
def update_table_and_selection(bounds, initial_bounds, geojson, _click_data,
                               reset_clicks, alvara_filter, name_filter,
                               address_filter, boteco_filter,
                               _marker_clicks, table_rows):

    ctx  = dash.callback_context
    trig = ctx.triggered_id

    # ───── 1) Clique em marcador → só seleciona  ──────────────────────
    if isinstance(trig, dict) and trig.get('type') == 'marker':
        nome = trig['index']
        if table_rows:
            sel = [i for i, r in enumerate(table_rows)
                   if r['NOME_FANTASIA'] == nome]
            return dash.no_update, sel
        return dash.no_update, []

    # ───── 2) Reset ou 1ª execução  →  usa apenas os bounds  ──────────
    if trig in (None, 'reset-btn'):
        bounds = bounds or initial_bounds
        lat_min, lon_min = bounds[0]
        lat_max, lon_max = bounds[1]
        rect = [lat_min, lon_min, lat_max, lon_max]

    # ───── 3) Filtros normais + eventual retângulo  ──────────────────
    else:
        # a) texto / checkboxes
        filtered = df.copy()
        if alvara_filter:
            filtered = filtered[filtered['IND_POSSUI_ALVARA']
                                .str.upper().isin(alvara_filter)]
        if isinstance(name_filter, str) and name_filter:
            filtered = filtered[filtered['NOME_FANTASIA']
                                .str.contains(name_filter, case=False, na=False)]
        if isinstance(address_filter, str) and address_filter:
            filtered = filtered[filtered['FULL_ADDRESS']
                                .str.contains(address_filter, case=False, na=False)]
        if boteco_filter == [1]:
            filtered = filtered[filtered['Comida de Boteco'] == 1]

        # b) retângulo desenhado ou, se não houver, bounds do mapa
        if geojson and geojson.get('features'):
            feat = next((f for f in reversed(geojson['features'])
                         if f.get('geometry') and f['geometry'].get('coordinates')), None)
            if feat:
                coords = feat['geometry']['coordinates'][0]
                lats   = [p[1] for p in coords]
                lons   = [p[0] for p in coords]
                rect   = [min(lats), min(lons), max(lats), max(lons)]
            else:
                bounds = bounds or initial_bounds
                lat_min, lon_min = bounds[0]
                lat_max, lon_max = bounds[1]
                rect = [lat_min, lon_min, lat_max, lon_max]
        else:
            bounds = bounds or initial_bounds
            lat_min, lon_min = bounds[0]
            lat_max, lon_max = bounds[1]
            rect = [lat_min, lon_min, lat_max, lon_max]

        # c) aplica corte espacial
        ids = []; range_search(kd_tree, rect, ids)
        filtered = filtered.loc[filtered.index.intersection(ids)]

        return filtered.to_dict('records'), []

    # (chega aqui só no caso “reset” ou primeira execução)
    ids = []; range_search(kd_tree, rect, ids)
    rows = df.loc[ids].to_dict('records')
    return rows, []



# ── mantém UM retângulo por vez ───────────────────────────
@app.callback(
    Output("edit_control", "geojson", allow_duplicate=True),
    [
        Input("edit_control", "geojson"), 
        Input("map", "clickData")          
    ],
    prevent_initial_call=True
)
def manage_rectangle(geojson, clickData):
    trig = callback_context.triggered_id

    # 1) Desenho novo ou edição → podar tudo menos o último
    if isinstance(trig, dict) and trig.get("prop_id", "").startswith("edit_control"):
        feats = geojson.get("features", [])
        if len(feats) > 1:
            return {"type": "FeatureCollection", "features": [feats[-1]]}
        raise PreventUpdate

    # 2) Clique no mapa → limpar completamente
    if isinstance(trig, dict) and trig.get("prop_id", "").startswith("map"):
        if geojson and geojson.get("features"):
            return {"type": "FeatureCollection", "features": []}
    raise PreventUpdate



# ───────────────────────────────────────────────────────────
# EXECUTA O SERVIDOR
# ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=8050)

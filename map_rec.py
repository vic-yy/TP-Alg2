import pandas as pd
import numpy as np
import dash
from dash import html, dash_table, dcc
import dash_leaflet as dl
from dash.dependencies import Input, Output, State
import re
import os
from pyproj import Transformer
import json

# Set dataset path to the same directory as the script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '20250401_bares_e_restaurantes.csv')

# Load dataset
df = pd.read_csv(data_path, encoding='utf-8', low_memory=False, sep=';')

# Convert UTM to latitude/longitude (assuming UTM Zone 23S, EPSG:31983)
transformer = Transformer.from_crs("EPSG:31983", "EPSG:4326", always_xy=True)
def utm_to_latlon(geometry):
    try:
        x, y = map(float, re.findall(r'POINT \(([\d.]+) ([\d.]+)\)', geometry)[0])
        lon, lat = transformer.transform(x, y)
        return lat, lon
    except:
        return np.nan, np.nan

# Apply coordinate conversion
df[['latitude', 'longitude']] = df['GEOMETRIA'].apply(utm_to_latlon).apply(pd.Series)
df = df.dropna(subset=['latitude', 'longitude'])

# Preprocess data
df['NOME_FANTASIA'] = df['NOME_FANTASIA'].fillna(df['NOME'])

# Create full address by combining address fields
df['FULL_ADDRESS'] = df.apply(
    lambda row: f"{row['DESC_LOGRADOURO']} {row['NOME_LOGRADOURO']}, {row['NUMERO_IMOVEL']}"
                f"{', ' + row['COMPLEMENTO'] if pd.notna(row['COMPLEMENTO']) else ''}, {row['NOME_BAIRRO']}",
    axis=1
)

# Convert date to datetime format
df['DATA_INICIO_ATIVIDADE'] = pd.to_datetime(df['DATA_INICIO_ATIVIDADE'], format='%d-%m-%Y', errors='coerce')
df['DATA_INICIO_STR'] = df['DATA_INICIO_ATIVIDADE'].dt.strftime('%Y-%m-%d')

df = df[['NOME_FANTASIA', 'latitude', 'longitude', 'FULL_ADDRESS', 'DATA_INICIO_ATIVIDADE', 'IND_POSSUI_ALVARA', 'DATA_INICIO_STR']]

# k-d Tree implementation
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
        if (rect[0] <= node.point[0] <= rect[2] and rect[1] <= node.point[1] <= rect[3]):
            result.append(node.data)
    
    if point_val >= min_val:
        range_search(node.left, rect, result, depth + 1)
    if point_val <= max_val:
        range_search(node.right, rect, result, depth + 1)

# Build k-d tree
points = df[['latitude', 'longitude']].values.tolist()
data = df.index.tolist()
kd_tree = build_kd_tree(points, data)

# Dash app setup
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# Custom JavaScript for marker hover and click
app.scripts.append_script({
    'external_url': 'https://unpkg.com/leaflet@1.7.1/dist/leaflet.js'
})

# Map and table layout
app.layout = html.Div([
    dcc.Store(id='initial-bounds', data=[[-20, -44], [-19, -43]]),
    dcc.Interval(id='startup-trigger', interval=100, n_intervals=0, max_intervals=1),
    dcc.Store(id='geojson-store'),

    # Left: Map
    html.Div([
        dl.Map(
            id='map',
            style={'width': '100%', 'height': '100%'},
            center=[-19.8800, -43.9378],
            zoom=12,
            children=[
                dl.TileLayer(),
                dl.FeatureGroup([
                    dl.EditControl(
                        id="edit_control",
                        draw={"rectangle": True, "polygon": False, "marker": False,
                              "circle": False, "polyline": False, "circlemarker": False},
                        edit={"edit": False}
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

    # Right: Table + Filter Buttons
    html.Div([
        
        # Buttons and Filters
        html.Div([
            # Row 1: Reset Button
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

            # Row 2: Nome Filter
            html.Div([
                html.Label('Nome:', style={'marginBottom': '0px', 'padding': '5px', 'width': '90px', 'font-weight': 'bold'}),
                dcc.Input(id='name-filter', type='text', placeholder='Buscar por nome...', debounce=True, style={
                    'width': '100%', 'height': '32px'
                })
            ], style={'marginBottom': '5px', 'display': 'flex', 'flexDirection': 'row'}),

            # Row 3: Endereço Filter
            html.Div([
                html.Label('Endereço:', style={'marginBottom': '0px', 'padding': '5px', 'width': '90px', 'font-weight': 'bold'}),
                dcc.Input(id='address-filter', type='text', placeholder='Buscar por endereço...', debounce=True, style={
                    'width': '100%', 'height': '32px'
                })
            ], style={'marginBottom': '5px', 'display': 'flex', 'flexDirection': 'row'}),

            # Row 4: Alvará Checkboxes
            html.Div([
                html.Label('Alvará:', style={'marginBottom': '0px', 'padding': '5px', 'width': '90px', 'font-weight': 'bold'}),
                dcc.Checklist(
                    id='alvara-filter',
                    options=[
                        {'label': 'Sim', 'value': 'SIM'},
                        {'label': 'Não', 'value': 'NÃO'}
                    ],
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
            ], style={'marginBottom': '5px', 'display': 'flex', 'flexDirection': 'row'})
        ]),

        # Scrollable Table Container
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
                style_table={
                    'height': '100%',
                    'overflowY': 'auto'
                },
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
                    {'if': {'row_index': 'odd'},
                    'backgroundColor': '#f0f0f0'},
                    {'if': {'state': 'selected'},
                    'backgroundColor': '#d3e0ea',
                    'border': '1px solid #0074D9'}
                ],
                style_header={
                    'textAlign': 'center',
                    'paddingLeft': '6px'
                },
            )
        ], style={
            'flex': '1',
            'overflow': 'auto'
        })
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
    'flexWrap': 'wrap',  # Enables responsiveness
    'gap': '0px',
    'height': '98vh',
    'width': '98vw',
    'padding': '0px',
    'boxSizing': 'border-box',
    'overflow': 'hidden',
    'backgroundColor': "#ffffff"
})


# Update markers on map
@app.callback(
    Output('markers', 'children'),
    [Input('map', 'bounds'),
     Input('initial-bounds', 'data'),
     Input('table', 'selected_rows')],
    [State('table', 'data')]
)
def update_markers(bounds, initial_bounds, selected_rows, table_data):
    bounds = bounds or initial_bounds
    if not bounds:
        return []
    lat_min, lon_min = bounds[0]
    lat_max, lon_max = bounds[1]
    rect = [lat_min, lon_min, lat_max, lon_max]
    
    result = []
    range_search(kd_tree, rect, result)
    
    filtered_df = df.loc[result]
    
    # Skip random sampling if a row is selected
    if not selected_rows or not table_data:
        if len(filtered_df) > 200:
            filtered_df = filtered_df.sample(n=200, random_state=None)
    
    # Get selected establishment from table
    selected_name = None
    if selected_rows and table_data:
        selected_name = table_data[selected_rows[0]]['NOME_FANTASIA']
    
    markers = []
    for _, row in filtered_df.iterrows():
        is_selected = row['NOME_FANTASIA'] == selected_name
        icon = {
            'iconUrl': 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
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


# callback to transfer data
@app.callback(
    Output('geojson-store', 'data'),
    Input('edit_control', 'geojson')
)
def sync_geojson(geojson):
    return geojson


# Handle table update and selection
@app.callback(
    [Output('table', 'data'), Output('table', 'selected_rows')],
    [
        Input('map', 'bounds'),
        Input('initial-bounds', 'data'),
        Input('edit_control', 'geojson'), 
        Input('reset-btn', 'n_clicks'),
        Input('alvara-filter', 'value'),
        Input('name-filter', 'value'),
        Input('address-filter', 'value'),
        Input({'type': 'marker', 'index': dash.dependencies.ALL}, 'n_clicks')
    ],
    [State('table', 'data')]
)
def update_table_and_selection(map_bounds, initial_bounds, geojson,
                               reset_clicks, alvara_filter, name_filter, address_filter, 
                               marker_clicks, table_data):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    if not ctx.triggered or 'startup-trigger' in triggered_id:
        bounds = map_bounds or initial_bounds
        if bounds:
            lat_min, lon_min = bounds[0]
            lat_max, lon_max = bounds[1]
            rect = [lat_min, lon_min, lat_max, lon_max]
            result = []
            range_search(kd_tree, rect, result)
            return df.loc[result].to_dict('records'), []
        else:
            return df.to_dict('records'), []

    triggered_id = ctx.triggered[0]["prop_id"]

    # Handle reset
    if 'reset-btn' in triggered_id:
        return df.to_dict('records'), []

    # Handle marker click
    if isinstance(ctx.triggered[0]["value"], dict) and ctx.triggered[0]["value"] is not None:
        marker_name = ctx.triggered[0]["value"].get("index")
        if table_data:
            for idx, row in enumerate(table_data):
                if row['NOME_FANTASIA'] == marker_name:
                    return dash.no_update, [idx]
        return dash.no_update, []

    # Filter by map bounds and Alvará checkbox
    filtered_df = df.copy()

    # === Fix: Ensure filters are strings ===
    name_filter = name_filter if isinstance(name_filter, str) else ''
    address_filter = address_filter if isinstance(address_filter, str) else ''

    # Apply Alvará filter (checkbox-based)
    if alvara_filter:
        filtered_df = filtered_df[filtered_df['IND_POSSUI_ALVARA'].str.upper().isin(alvara_filter)]

    # Apply name substring filter
    if name_filter:
        filtered_df = filtered_df[filtered_df['NOME_FANTASIA'].str.contains(name_filter, case=False, na=False)]

    # Apply address substring filter
    if address_filter:
        filtered_df = filtered_df[filtered_df['FULL_ADDRESS'].str.contains(address_filter, case=False, na=False)]

    # Check if a valid rectangle has been drawn
    has_rectangle = geojson and geojson.get("features") and any(
        f.get("geometry") and f["geometry"].get("coordinates") for f in geojson["features"]
    )

    # Apply spatial filtering
    if has_rectangle:
        feature = next((f for f in geojson["features"]
                        if f.get("geometry") and f["geometry"].get("coordinates")), None)
        if feature:
            coords = feature["geometry"]["coordinates"][0]
            lons = [p[0] for p in coords]
            lats = [p[1] for p in coords]
            lat_min, lat_max = min(lats), max(lats)
            lon_min, lon_max = min(lons), max(lons)
            rect = [lat_min, lon_min, lat_max, lon_max]
            result = []
            range_search(kd_tree, rect, result)
            valid_indices = filtered_df.index.intersection(result)
            filtered_df = filtered_df.loc[valid_indices]
    else:
        # No rectangle → use visible map bounds
        bounds = map_bounds or initial_bounds
        if bounds:
            lat_min, lon_min = bounds[0]
            lat_max, lon_max = bounds[1]
            rect = [lat_min, lon_min, lat_max, lon_max]
            result = []
            range_search(kd_tree, rect, result)
            valid_indices = filtered_df.index.intersection(result)
            filtered_df = filtered_df.loc[valid_indices]

    return filtered_df.to_dict('records'), []


# Run the Dash server
if __name__ == '__main__':
    app.run(debug=True, port=8050)
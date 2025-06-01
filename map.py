import pandas as pd
import numpy as np
import dash
from dash import html, dash_table
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
df = df[['NOME_FANTASIA', 'latitude', 'longitude', 'FULL_ADDRESS', 'DATA_INICIO_ATIVIDADE', 'IND_POSSUI_ALVARA']]

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
    # Left: Map
    html.Div([
        dl.Map(
            id='map',
            style={'width': '100%', 'height': '100%'},
            center=[-19.9208, -43.9378],
            zoom=12,
            children=[
                dl.TileLayer(),
                dl.Rectangle(id='selector', bounds=[[0, 0], [0, 0]], color='blue', fillOpacity=0.2),
                dl.LayerGroup(id='markers')
            ]
        )
    ], style={
        'flex': '1',
        'minWidth': '50%',
        'padding': '10px',
        'boxSizing': 'border-box',
        'height': '100%'
    }),

    # Right: Table + Button
    html.Div([
        html.Button('Reset Filter', id='reset-btn', style={
            'marginBottom': '10px',
            'width': '100%',
            'padding': '0px',
            'backgroundColor': '#0074D9',
            'color': 'white',
            'border': 'none',
            'borderRadius': '0px',
            'cursor': 'pointer',
            'textAlign': 'center'
        }),

        # Scrollable Table Container
        html.Div([
            dash_table.DataTable(
                id='table',
                columns=[
                    {'name': 'Nome', 'id': 'NOME_FANTASIA'},
                    {'name': 'Endereço', 'id': 'FULL_ADDRESS'},
                    {'name': 'Data Início', 'id': 'DATA_INICIO_ATIVIDADE'},
                    {'name': 'Possui Alvará', 'id': 'IND_POSSUI_ALVARA'}
                ],
                page_size=15,
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
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f0f0f0'
                    },
                    {
                        'if': {'state': 'selected'},
                        'backgroundColor': '#d3e0ea',
                        'border': '1px solid #0074D9'
                    }
                ],
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
    [Input('map', 'bounds'), Input('table', 'selected_rows')],
    [State('table', 'data')]
)
def update_markers(bounds, selected_rows, table_data):
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
            'iconUrl': 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png' if is_selected else 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
            'iconSize': [38, 55] if is_selected else [25, 41],
            'iconAnchor': [19, 95] if is_selected else [12, 41],  # Fix red pin at bottom
            'popupAnchor': [1, -34],
            'shadowSize': [41, 41]
        }
        markers.append(
            dl.Marker(
                position=[row['latitude'], row['longitude']],
                icon=icon,
                children=[
                    dl.Tooltip(row['NOME_FANTASIA']),
                    dl.Popup(html.Div([
                        html.H4(row['NOME_FANTASIA'], style={'fontSize': '16px', 'margin': '0'}),
                        html.P(row['FULL_ADDRESS'], style={'fontSize': '14px', 'margin': '5px 0'})
                    ]))
                ],
                id={'type': 'marker', 'index': row['NOME_FANTASIA']}
            )
        )
    return markers

# Handle table update and selection
@app.callback(
    [Output('table', 'data'), Output('selector', 'bounds'), Output('table', 'selected_rows')],
    [Input('map', 'bounds'), Input('reset-btn', 'n_clicks'), Input({'type': 'marker', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State('selector', 'bounds'), State('table', 'data')]
)
def update_table_and_selection(map_bounds, n_clicks_reset, marker_clicks, current_bounds, table_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return df.to_dict('records'), [[0, 0], [0, 0]], []
    
    if ctx.triggered[0]['prop_id'] == 'reset-btn.n_clicks':
        return df.to_dict('records'), [[0, 0], [0, 0]], []
    
    triggered_id = ctx.triggered[0]['prop_id']
    if 'marker' in triggered_id:
        marker_name = json.loads(triggered_id.split('.')[0])['index']
        if table_data:
            for idx, row in enumerate(table_data):
                if row['NOME_FANTASIA'] == marker_name:
                    return dash.no_update, dash.no_update, [idx]
        return dash.no_update, dash.no_update, []
    
    if map_bounds:
        lat_min, lon_min = map_bounds[0]
        lat_max, lon_max = map_bounds[1]
        rect = [lat_min, lon_min, lat_max, lon_max]
        result = []
        range_search(kd_tree, rect, result)
        filtered_df = df.loc[result]
        return filtered_df.to_dict('records'), [[0, 0], [0, 0]], []
    
    return df.to_dict('records'), [[0, 0], [0, 0]], []

# Run the Dash server
if __name__ == '__main__':
    app.run(debug=True, port=8050)
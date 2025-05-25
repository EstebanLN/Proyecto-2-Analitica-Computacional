import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

# Cargar datos CSV
df = pd.read_csv("Clean_data.csv")

# Inicializar la app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],    suppress_callback_exceptions=True )
app.title = "Resultados Saber 11"

# Layout con Tabs
app.layout = dbc.Container([
    html.Div(
        [
            html.Div(
                html.H1("Dashboard Saber 11", style={
                    "fontWeight": "bold",
                    "margin": "0",
                    "color": "white"
                }),
                style={"flex": "1"}
            ),
            html.Img(
                src="/assets/icfes.png",  # Asegúrate de que el archivo esté en la carpeta assets/
                style={"height": "60px"}
            )
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "padding": "20px",
            "backgroundColor": "#003366",
            "borderRadius": "8px",
            "boxShadow": "0px 4px 6px rgba(0,0,0,0.2)"
        }
    ),
    
    dcc.Tabs(id="tabs", value="tab-visual", children=[
        dcc.Tab(label="📊 Visualización de Datos", value="tab-visual"),
        dcc.Tab(label="🤖 Modelo Predictivo", value="tab-modelo"),
    ]),
    html.Div(id="tabs-content")
], fluid=True)


# Callback para actualizar el contenido según la pestaña seleccionada
@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "value")
)
def render_tab_content(tab):
    if tab == "tab-visual":
        return html.Div([
            html.H4("Distribución del Puntaje Global por Categoría"),
            dcc.Dropdown(
                id="dropdown-variable",
                options=[{"label": var, "value": var} for var in [
                    "cole_area_ubicacion",
                    "fami_estratovivienda",
                    "cole_jornada",
                    "estu_genero",
                    "fami_tieneinternet",
                    "fami_tienecomputador"
                ]],
                value="fami_estratovivienda",
                style={"width": "50%"}
            ),
            dcc.Graph(id="boxplot-output")
        ])
    elif tab == "tab-modelo":
        return html.Div([
            html.H4("Predicción con Modelo (en construcción)"),
            html.P("Aquí irá el formulario para ingresar datos del estudiante y obtener su predicción.")
        ])

# Callback para actualizar gráfico en la pestaña de visualización
@app.callback(
    Output("boxplot-output", "figure"),
    Input("dropdown-variable", "value")
)
def update_graph(var):
    fig = px.box(df, x=var, y="punt_global", title=f"Puntaje global según {var}")
    return fig

# Ejecutar app
if __name__ == "__main__":
    app.run(debug=True)

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

# Cargar datos
df = pd.read_csv("tus_datos.csv")  # Cambia por tu archivo

# Inicializar la app con multipage support
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Diseño base con navegación
app.layout = dbc.Container([
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Visualizaciones", href="/visualizaciones")),
            dbc.NavItem(dbc.NavLink("Modelo (próximamente)", href="/modelo")),
        ],
        brand="Dashboard Saber 11",
        color="primary",
        dark=True,
    ),
    dash.page_container
])

# ------------------ Página de Visualizaciones ------------------

dash.register_page("visualizaciones", path="/visualizaciones")

layout_visualizaciones = dbc.Container([
    html.H2("Exploración de Puntaje Global"),
    dcc.Dropdown(
        id="dropdown-variable",
        options=[{"label": var, "value": var} for var in ["cole_area_ubicacion", "fami_estratovivienda", "cole_jornada"]],
        value="fami_estratovivienda"
    ),
    dcc.Graph(id="boxplot-output")
])

@dash.callback(
    Output("boxplot-output", "figure"),
    Input("dropdown-variable", "value")
)
def update_boxplot(var):
    fig = px.box(df, x=var, y="punt_global", title=f"Puntaje global según {var}")
    return fig

# ------------------ Página futura del modelo ------------------

dash.register_page("modelo", path="/modelo")

layout_modelo = dbc.Container([
    html.H2("Predicción con Modelo (en construcción)"),
    html.P("Aquí irá el formulario para ingresar datos del estudiante y obtener su predicción.")
])

# ------------------ Ejecutar ------------------

if __name__ == "__main__":
    app.run(debug=True)


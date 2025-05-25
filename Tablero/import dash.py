import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import json

# Cargar datos
df = pd.read_csv("Clean_data.csv")

# Inicializar app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Resultados Saber 11"

# Layout principal
app.layout = dbc.Container([
    html.Div([
        html.Div(
            html.H1("Resultados Saber 11", style={"margin": "0", "color": "#003366", "fontWeight": "normal"}),
            style={"flex": "1"}
        ),
        html.Img(src="/assets/icfes.png", style={"height": "60px"})
    ], style={
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "space-between",
        "padding": "20px",
        "backgroundColor": "#e6f0fa",
        "borderRadius": "8px",
        "boxShadow": "0px 4px 6px rgba(0,0,0,0.1)"
    }),

    html.Div([
        html.P("En esta aplicación, podrás explorar los resultados de un estudio realizado por un grupo de estudiantes de la clase de analítica computacional para la toma de decisiones. Hemos diseñado esta plataforma para que puedas acceder a un análisis detallado de los resultados del ICFES 2018 en Colombia, permitiéndote comprender mejor las variables que influyen en el desempeño académico de los estudiantes."),
        html.P("Además, tendrás la oportunidad de ingresar información de un nuevo estudiante en nuestra sección de predicciones. A través de este proceso, podrás evaluar su posible desempeño en el examen ICFES con base en las tendencias observadas en los datos existentes."),
        html.P("Te invitamos a navegar por las diferentes secciones de la aplicación y descubrir todo lo que tenemos para ofrecerte.")
    ], style={
        "backgroundColor": "#f8fbff",
        "padding": "20px",
        "marginTop": "20px",
        "marginBottom": "20px",
        "borderRadius": "8px",
        "color": "#003366",
        "fontSize": "16px",
        "lineHeight": "1.6"
    }),

    dcc.Tabs(
        id="tabs",
        value="tab-visual",
        children=[
            dcc.Tab(label="Visualización de Datos", value="tab-visual", style={
                "backgroundColor": "#e6f0fa", "color": "#003366", "fontWeight": "normal", "padding": "10px"},
                selected_style={"backgroundColor": "#003366", "color": "white", "fontWeight": "bold", "padding": "10px"}
            ),
            dcc.Tab(label="Predicciones", value="tab-modelo", style={
                "backgroundColor": "#e6f0fa", "color": "#003366", "fontWeight": "normal", "padding": "10px"},
                selected_style={"backgroundColor": "#003366", "color": "white", "fontWeight": "bold", "padding": "10px"}
            ),
        ]
    ),
    html.Div(id="tabs-content")
], fluid=True)

@app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_tab_content(tab):
    if tab == "tab-visual":
        return html.Div([
            html.H4("Distribución del Puntaje Global Promedio"),

            html.Div([
                html.Div([dcc.Graph(id="grafico-estrato")], style={"width": "50%", "display": "inline-block", "padding": "10px"}),
                html.Div([dcc.Graph(id="grafico-genero")], style={"width": "50%", "display": "inline-block", "padding": "10px"})
            ]),
            html.Div([dcc.Graph(id="grafico-mapa")], style={"padding": "10px"})
        ])
    elif tab == "tab-modelo":
        return html.Div([
            html.H4("Predicción con Modelo (en construcción)"),
            html.P("Aquí irá el formulario para ingresar datos del estudiante y obtener su predicción.")
        ])

@app.callback(Output("grafico-estrato", "figure"), Input("tabs", "value"))
def actualizar_grafico_estrato(_):
    df_avg = df.groupby("fami_estratovivienda", as_index=False)["punt_global"].mean().round(2)
    fig = px.bar(df_avg, x="fami_estratovivienda", y="punt_global",
                 labels={"fami_estratovivienda": "Estrato", "punt_global": "Puntaje Promedio"},
                 title="Promedio del Puntaje Global según Estrato",
                 color="fami_estratovivienda",
                 color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_layout(plot_bgcolor="#e6f0fa", paper_bgcolor="#e6f0fa", font=dict(color="#003366"))
    return fig

@app.callback(Output("grafico-genero", "figure"), Input("tabs", "value"))
def actualizar_grafico_genero(_):
    df_avg = df.groupby("estu_genero", as_index=False)["punt_global"].mean().round(2)
    fig = px.bar(df_avg, x="estu_genero", y="punt_global",
                 labels={"estu_genero": "Género", "punt_global": "Puntaje Promedio"},
                 title="Promedio del Puntaje Global según Género",
                 color="estu_genero",
                 color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_layout(plot_bgcolor="#e6f0fa", paper_bgcolor="#e6f0fa", font=dict(color="#003366"))
    return fig

@app.callback(Output("grafico-mapa", "figure"), Input("tabs", "value"))
def actualizar_mapa(_):
    df_map = df.groupby("estu_depto_reside", as_index=False).agg(
        punt_global=("punt_global", "mean"),
        num_estudiantes=("punt_global", "count")
    )
    df_map["punt_global"] = df_map["punt_global"].round(3)

    with open("colombia_departamentos.json", encoding="utf-8") as f:
        geojson_colombia = json.load(f)

    fig = px.choropleth(df_map, geojson=geojson_colombia, locations="estu_depto_reside",
                        featureidkey="properties.NOMBRE_DPT",
                        color="punt_global",
                        color_continuous_scale="Blues",
                        hover_name="estu_depto_reside",
                        hover_data={"punt_global": True, "num_estudiantes": True, "estu_depto_reside": False},
                        labels={"estu_depto_reside": "Departamento", "punt_global": "Puntaje Promedio", "num_estudiantes": "N. Estudiantes"},
                        title="Puntaje Global Promedio por Departamento")
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, paper_bgcolor="#f8fbff")
    return fig

# Ejecutar
if __name__ == "__main__":
    app.run(debug=True)

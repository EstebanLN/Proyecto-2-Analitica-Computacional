import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import json
<<<<<<< HEAD
import numpy as np
from keras.models import load_model

# Cargar modelo y columnas usadas
modelo = load_model("modelo_clasificacion.keras")
columnas_modelo = np.load("columnas_modelo.npy", allow_pickle=True)
=======
>>>>>>> 23f15c1ec0fbb7d345af3e47623ce26ec9bb28b4

# Cargar datos
df = pd.read_csv("Clean_data.csv")

<<<<<<< HEAD
# Variables utilizadas por el modelo (en orden)
variables_modelo = [
    "cole_area_ubicacion", "cole_bilingue", "cole_calendario", "cole_caracter",
    "cole_genero", "cole_jornada", "cole_naturaleza", "cole_depto_ubicacion",
    "cole_mcpio_ubicacion", "estu_tipodocumento", "estu_genero", "estu_depto_reside",
    "estu_mcpio_reside", "estu_nacionalidad", "estu_pais_reside", "estu_privado_libertad",
    "fami_cuartoshogar", "fami_educacionmadre", "fami_educacionpadre", "fami_estratovivienda",
    "fami_personashogar", "fami_tieneautomovil", "fami_tienecomputador", "fami_tieneinternet",
    "fami_tienelavadora"
]

etiquetas = {v: v.replace("_", " ").capitalize() for v in variables_modelo}

=======
>>>>>>> 23f15c1ec0fbb7d345af3e47623ce26ec9bb28b4
# Inicializar app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Resultados Saber 11"

# Layout principal
app.layout = dbc.Container([
    html.Div([
        html.Div(
            html.H1("Resultados Saber 11-2018", style={"margin": "0", "color": "#003366", "fontWeight": "normal"}),
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
<<<<<<< HEAD
        html.P("En esta aplicación, podrás explorar los resultados del ICFES 2018 en Colombia."),
        html.P("Podrás ingresar información de un nuevo estudiante y estimar su puntaje."),
    ], style={"backgroundColor": "#f8fbff", "padding": "20px", "marginTop": "20px", "marginBottom": "20px", "borderRadius": "8px", "color": "#003366", "fontSize": "16px", "lineHeight": "1.6"}),
=======
        html.P("En esta aplicación, podrás explorar los resultados de un estudio realizado por un grupo de estudiantes de la clase de analítica computacional para la toma de decisiones. Hemos diseñado esta plataforma para que puedas acceder a un análisis detallado de los resultados del ICFES 2018 en Colombia, permitiéndote comprender mejor las variables que influyen en el desempeño académico de los estudiantes."),
        html.P("Además, tendrás la oportunidad de ingresar información de un nuevo estudiante en nuestra sección de entrada de datos. A través de este proceso, podrás evaluar su posible desempeño en el examen ICFES con base en las tendencias observadas en los datos existentes."),
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
>>>>>>> 23f15c1ec0fbb7d345af3e47623ce26ec9bb28b4

    dcc.Tabs(
        id="tabs",
        value="tab-visual",
        children=[
            dcc.Tab(label="Visualización de Datos", value="tab-visual", style={
                "backgroundColor": "#e6f0fa", "color": "#003366", "fontWeight": "normal", "padding": "10px"},
                selected_style={"backgroundColor": "#003366", "color": "white", "fontWeight": "bold", "padding": "10px"}
            ),
            dcc.Tab(label="Entrada de Datos", value="tab-entrada", style={
                "backgroundColor": "#e6f0fa", "color": "#003366", "fontWeight": "normal", "padding": "10px"},
                selected_style={"backgroundColor": "#003366", "color": "white", "fontWeight": "bold", "padding": "10px"}
            ),
            dcc.Tab(label="Predicción", value="tab-modelo", style={
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
            html.H4("Factores Socioeconómicos, Demográficos y Espaciales"),
            html.Div([
                html.Div([dcc.Graph(id="grafico-estrato")], style={"width": "50%", "display": "inline-block", "padding": "10px"}),
                html.Div([dcc.Graph(id="grafico-genero")], style={"width": "50%", "display": "inline-block", "padding": "10px"})
            ]),
            html.Div([dcc.Graph(id="grafico-mapa")], style={"padding": "10px"})
        ])
    elif tab == "tab-entrada":
        return html.Div([
            html.H4("Formulario de Entrada de Datos", style={"color": "#003366"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Área de ubicación del colegio:"),
                    dcc.Dropdown(options=[{"label": i, "value": i} for i in ["URBANO", "RURAL"]], value="URBANO")
                ]),
                dbc.Col([
                    html.Label("¿El colegio es bilingüe?"),
                    dcc.Dropdown(options=[{"label": i, "value": i} for i in ["Sí", "No", "Desconocido"]], value="Desconocido")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Calendario del colegio:"),
                    dcc.Dropdown(options=[{"label": i, "value": i} for i in ["A", "B", "OTRO"]], value="A")
                ]),
                dbc.Col([
                    html.Label("Carácter del colegio:"),
                    dcc.Dropdown(options=[{"label": i, "value": i} for i in ["ACADÉMICO", "TÉCNICO", "OTRO"]], value="ACADÉMICO")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Jornada del colegio:"),
                    dcc.Dropdown(options=[{"label": i, "value": i} for i in ["MAÑANA", "TARDE", "ÚNICA", "NOCHE"]], value="ÚNICA")
                ]),
                dbc.Col([
                    html.Label("Naturaleza del colegio:"),
                    dcc.Dropdown(options=[{"label": i, "value": i} for i in ["OFICIAL", "NO OFICIAL"]], value="OFICIAL")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Selecciona el género del estudiante:"),
                    dcc.Dropdown(options=[{"label": i, "value": i} for i in ["M", "F", "Desconocido"]], value="Desconocido")
                ]),
                dbc.Col([
                    html.Label("Selecciona la nacionalidad del estudiante:"),
                    dcc.Dropdown(options=[{"label": "Colombia", "value": "Colombia"}, {"label": "Otra", "value": "Otra"}], value="Colombia")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Nivel educativo de la madre:"),
                    dcc.Dropdown(options=[{"label": i, "value": i} for i in ["Primaria", "Secundaria", "Técnico", "Universitario", "Desconocido"]], value="Desconocido")
                ]),
                dbc.Col([
                    html.Label("Nivel educativo del padre:"),
                    dcc.Dropdown(options=[{"label": i, "value": i} for i in ["Primaria", "Secundaria", "Técnico", "Universitario", "Desconocido"]], value="Desconocido")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Estrato de la vivienda:"),
                    dcc.Dropdown(options=[{"label": str(i), "value": str(i)} for i in range(1, 7)] + [{"label": "Desconocido", "value": "Desconocido"}], value="Desconocido")
                ]),
                dbc.Col([
                    html.Label("Número de personas en el hogar:"),
                    dcc.Dropdown(options=[{"label": str(i), "value": str(i)} for i in range(1, 11)] + [{"label": "Desconocido", "value": "Desconocido"}], value="Desconocido")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("¿La familia tiene automóvil?"),
                    dcc.Dropdown(options=[{"label": i, "value": i} for i in ["Sí", "No", "Desconocido"]], value="Desconocido")
                ]),
                dbc.Col([
                    html.Label("¿La familia tiene computador?"),
                    dcc.Dropdown(options=[{"label": i, "value": i} for i in ["Sí", "No", "Desconocido"]], value="Desconocido")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("¿La familia tiene internet?"),
                    dcc.Dropdown(options=[{"label": i, "value": i} for i in ["Sí", "No", "Desconocido"]], value="Desconocido")
                ]),
                dbc.Col([
                    html.Label("¿La familia tiene lavadora?"),
                    dcc.Dropdown(options=[{"label": i, "value": i} for i in ["Sí", "No", "Desconocido"]], value="Desconocido")
                ])
            ])
        ], style={"padding": "20px", "backgroundColor": "#e6f0fa", "color": "#003366"})
    elif tab == "tab-modelo":
        return html.Div([
            html.H4("Predicción con Modelo (en construcción)"),
            html.P("Aquí irá el modelo predictivo para estimar el puntaje global.")
        ])

@app.callback(Output("grafico-estrato", "figure"), Input("tabs", "value"))
def actualizar_grafico_estrato(_):
    df_avg = df.groupby("fami_estratovivienda", as_index=False)["punt_global"].mean().round(2)
    fig = px.bar(df_avg, x="fami_estratovivienda", y="punt_global",
                 labels={"fami_estratovivienda": "Estrato", "punt_global": "Puntaje Promedio"},
                 title="Promedio del Puntaje Global según Estrato",
                 color="fami_estratovivienda", color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_layout(plot_bgcolor="#e6f0fa", paper_bgcolor="#e6f0fa", font=dict(color="#003366"))
    return fig

@app.callback(Output("grafico-genero", "figure"), Input("tabs", "value"))
def actualizar_grafico_genero(_):
    df_filtrado = df[df["estu_genero"].notna() & df["punt_global"].notna()]
    fig = px.violin(df_filtrado, x="estu_genero", y="punt_global", box=True, points=False,
                    color="estu_genero", color_discrete_sequence=px.colors.sequential.Blues_r,
                    title="Distribución del Puntaje Global por Género")
    fig.update_layout(xaxis_title="Género", yaxis_title="Puntaje Global",
                      plot_bgcolor="#e6f0fa", paper_bgcolor="#e6f0fa",
                      font_color="#003366", legend_title="Género")
    return fig

@app.callback(Output("grafico-mapa", "figure"), Input("tabs", "value"))
def actualizar_mapa(_):
    df["estu_depto_reside"] = df["estu_depto_reside"].replace({"BOGOTA": "SANTAFE DE BOGOTA D.C"})
    df_map = df.groupby("estu_depto_reside", as_index=False).agg(punt_global=("punt_global", "mean"), num_estudiantes=("punt_global", "count"))
    df_map["punt_global"] = df_map["punt_global"].round(3)
    with open("colombia_departamentos.json", encoding="utf-8") as f:
        geojson_colombia = json.load(f)
    fig = px.choropleth(df_map, geojson=geojson_colombia, locations="estu_depto_reside",
                        featureidkey="properties.NOMBRE_DPT", color="punt_global",
                        color_continuous_scale="Blues", hover_name="estu_depto_reside",
                        hover_data={"punt_global": True, "num_estudiantes": True, "estu_depto_reside": False},
                        labels={"estu_depto_reside": "Departamento", "punt_global": "Puntaje Promedio", "num_estudiantes": "N. Estudiantes"},
                        title="Puntaje Global Promedio por Departamento")
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, paper_bgcolor="#f8fbff")
    return fig

<<<<<<< HEAD
@app.callback(Output("prediccion-output", "children"), Input("btn-predict", "n_clicks"), [State(var, "value") for var in variables_modelo])
def predecir(_, *valores):
    if None in valores:
        return "Por favor complete todos los campos para realizar la predicción."
    df_input = pd.DataFrame([valores], columns=variables_modelo)
    df_union = pd.concat([df[variables_modelo], df_input], axis=0)
    df_encoded = pd.get_dummies(df_union)
    df_encoded = df_encoded.reindex(columns=columnas_modelo, fill_value=0)
    fila_pred = df_encoded.tail(1).astype(float)
    puntaje = float(modelo.predict(fila_pred)[0][0])
    promedio = df["punt_global"].mean()
    categoria = "Superior al promedio" if puntaje >= promedio else "Inferior al promedio"
    return f"Categoría: {categoria}"

=======
# Ejecutar
>>>>>>> 23f15c1ec0fbb7d345af3e47623ce26ec9bb28b4
if __name__ == "__main__":
    app.run(debug=True)


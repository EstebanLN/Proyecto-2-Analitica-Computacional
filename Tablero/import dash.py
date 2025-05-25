import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import json

# Cargar datos CSV
df = pd.read_csv("Clean_data.csv")

# Inicializar app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Resultados Saber 11"

# Layout
app.layout = dbc.Container([
    # ENCABEZADO
    html.Div([
        html.Div(
            html.H1("Resultados Saber 11", style={
                "margin": "0",
                "color": "#003366",
                "fontWeight": "normal"
            }),
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

    # DESCRIPCIÓN
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

    # TABS
    dcc.Tabs(
        id="tabs",
        value="tab-visual",
        children=[
            dcc.Tab(
                label="Visualización de Datos",
                value="tab-visual",
                style={
                    "backgroundColor": "#e6f0fa",
                    "color": "#003366",
                    "fontWeight": "normal",
                    "padding": "10px"
                },
                selected_style={
                    "backgroundColor": "#003366",
                    "color": "white",
                    "fontWeight": "bold",
                    "padding": "10px"
                }
            ),
            dcc.Tab(
                label="Predicciones",
                value="tab-modelo",
                style={
                    "backgroundColor": "#e6f0fa",
                    "color": "#003366",
                    "fontWeight": "normal",
                    "padding": "10px"
                },
                selected_style={
                    "backgroundColor": "#003366",
                    "color": "white",
                    "fontWeight": "bold",
                    "padding": "10px"
                }
            ),
        ]
    ),

    html.Div(id="tabs-content")
], fluid=True)


# CALLBACK: contenido según pestaña
@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "value")
)
def render_tab_content(tab):
    if tab == "tab-visual":
        return html.Div([
            html.H4("Distribución del Puntaje Global Promedio por Categoría"),
            dcc.Dropdown(
                id="dropdown-variable",
                options=[
                    {"label": "Área del Colegio", "value": "cole_area_ubicacion"},
                    {"label": "Estrato de Vivienda", "value": "fami_estratovivienda"},
                    {"label": "Jornada del Colegio", "value": "cole_jornada"},
                    {"label": "Género del Estudiante", "value": "estu_genero"},
                    {"label": "Tiene Internet en el Hogar", "value": "fami_tieneinternet"},
                    {"label": "Departamento de Residencia", "value": "estu_depto_reside"}
                ],
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


# CALLBACK: gráfico según variable
@app.callback(
    Output("boxplot-output", "figure"),
    Input("dropdown-variable", "value")
)
def update_graph(var):
    if var == "estu_depto_reside":
        df_map = (
            df.groupby("estu_depto_reside", as_index=False)
            .agg(
                punt_global=("punt_global", "mean"),
                num_estudiantes=("punt_global", "count")
            )
        )
        df_map["punt_global"] = df_map["punt_global"].round(3)

        with open("colombia_departamentos.json", encoding="utf-8") as f:
            geojson_colombia = json.load(f)

        fig = px.choropleth(
            df_map,
            geojson=geojson_colombia,
            locations="estu_depto_reside",
            featureidkey="properties.NOMBRE_DPT",
            color="punt_global",
            color_continuous_scale="Blues",
            hover_name="estu_depto_reside",
            hover_data={
                "punt_global": True,
                "num_estudiantes": True,
                "estu_depto_reside": False
            },
            labels={
                "estu_depto_reside": "Departamento",
                "punt_global": "Puntaje Promedio",
                "num_estudiantes": "N. Estudiantes"
            },
            title="Puntaje Global Promedio por Departamento"
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
        return fig
    else:
        axis_titles = {
            "cole_area_ubicacion": "Área del Colegio",
            "fami_estratovivienda": "Estrato de Vivienda",
            "cole_jornada": "Jornada del Colegio",
            "estu_genero": "Género del Estudiante",
            "fami_tieneinternet": "Tiene Internet en el Hogar"
        }

        # Calcular promedios
        df_avg = df.groupby(var, as_index=False)["punt_global"].mean()
        df_avg["punt_global"] = df_avg["punt_global"].round(2)

        # Gráfico de barras
        fig = px.bar(
            df_avg,
            x=var,
            y="punt_global",
            title=f"Promedio del Puntaje Global según {axis_titles.get(var, var)}",
            labels={
                var: axis_titles.get(var, var),
                "punt_global": "Puntaje Promedio"
            },
            color=var, 
        color_discrete_sequence=px.colors.sequential.Blues_r 
        )

        fig.update_layout(
            plot_bgcolor="#e6f0fa",
            paper_bgcolor="#e6f0fa",
            font=dict(color="#003366"),
            title_font=dict(color="#003366")
        )

        return fig


# Ejecutar
if __name__ == "__main__":
    app.run(debug=True)

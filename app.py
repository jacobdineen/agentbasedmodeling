import time
import importlib
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Input, Output, State
import utils.dash_reusable_components as drc


app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
server = app.server


    
app.layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
        html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    children=[
                        # Change App Name here
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "Agent based modeling",
                                    href="https://github.com/plotly/dash-svm",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                    },
                                )
                            ],
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    # className="row",
                    children=[
                        html.Div(
                            # className="three columns",
                            id="left-column",
                            children=[
                                drc.Card(
                                    id="first-card",
                                        children=[
                                        drc.NamedSlider(
                                            name="Number of Agents",
                                            id="slider-dataset-sample-size",
                                            min=5,
                                            max=100,
                                            step=5,
                                            marks={
                                                str(i): str(i)
                                                for i in [5, 10, 25, 50, 100]
                                            },
                                            value=5,
                                            ),
                                        drc.NamedSlider(
                                            name="P(edge) - Random Graph Generation",
                                            id="slider-dataset-randomgraph",
                                            min=0,
                                            max=1,
                                            marks={
                                                i / 10: str(i / 10)
                                                for i in range(0, 11, 2)
                                            },
                                            step=0.1,
                                            value=0.5,
                                        ),
                                        drc.NamedSlider(
                                            name="P(selfish) - Distribution of Agent Strategies",
                                            id="slider-dataset-agentprofiles",
                                            min=0,
                                            max=1,
                                            marks={
                                                i / 10: str(i / 10)
                                                for i in range(0, 11, 2)
                                            },
                                            step=0.1,
                                            value=0.5,
                                        ),
                                        drc.NamedSlider(
                                            name="P* - Sampling Densities",
                                            id="slider-dataset-Sampling-Densities",
                                            min=0,
                                            max=1,
                                            marks={
                                                i / 10: str(i / 10)
                                                for i in range(0, 11, 2)
                                            },
                                            step=0.1,
                                            value=0.5,
                                        ),
                                        
                                    ],
                                ),
                                drc.Card(
                                    id="button-card",
                                    children=[
                                        drc.NamedSlider(
                                            name="Threshold",
                                            id="slider-threshold",
                                            min=0,
                                            max=1,
                                            value=0.5,
                                            step=0.01,
                                        ),
                                    ],
                                ),
                        html.Div(
                            id="div-graphs",
                            children=dcc.Graph(
                                id="graph-sklearn-svm",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                    )
                                ),
                            ),
                        ),
                    ],
                )
            ],
        ),
    ],
),
    ],
)

#https://community.plot.ly/t/loading-pandas-dataframe-into-data-table-through-a-callback/19354/16
'''
Want this to take slider values as inputs into env.play and display a df and other graphs

'''

# @app.callback(
#     Output("left-column", "children"),
#     [Input("slider-dataset-sample-size", "value"), Input("slider-dataset-randomgraph", "value")],
# )
# def update_output_div(input_value1, input_value2):
#     return 'You\'ve entered "{} and {}"'.format(str(input_value1), str(input_value2))



# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
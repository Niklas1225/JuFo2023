import os
import json
import dash
from dash import dcc
from dash import html
import dash_colorscales as dcs
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from PIL import Image
import numpy as np
import plotly.express as px
from mni import create_mesh_data, default_colorscale
from ssimClass import slice_img


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.title = "Brain Regions Viewer"

server = app.server

default_colorscale_index = [ea[1] for ea in default_colorscale]

MIN_SLIDER_VALUE = 0
MAX_SLIDER_VALUE = 300
SAVED_Z = 300#MAX_SLIDER_VALUE

axis_template = {
    "showbackground": True,
    "backgroundcolor": "#141414",
    "gridcolor": "rgb(255, 255, 255)",
    "zerolinecolor": "rgb(255, 255, 255)",
}

plot_layout = {
    "title": "",
    "margin": {"t": 0, "b": 0, "l": 0, "r": 0},
    "font": {"size": 12, "color": "white"},
    "showlegend": False,
    "plot_bgcolor": "#141414",
    "paper_bgcolor": "#141414",
    "scene": {
        "xaxis": axis_template,
        "yaxis": axis_template,
        "zaxis": axis_template,
        "aspectratio": {"x": 1, "y": 1.2, "z": 1},
        "camera": {"eye": {"x": 1.25, "y": 1.25, "z": 1.25}},
        "annotations": [],
    },
}

imshow_layout = {
    "paper_bgcolor": 'rgba(0,0,0,0)',
    "plot_bgcolor": 'rgba(0,0,0,0)',
}

button_style_dict = {
    "width": "300px",
    "height": "auto",
    'margin-bottom': '10px',
    "display": "block",
    "margin-left": "auto",
    "margin-right": "auto",
    #"color": "white",
    #"border-color": "white",
    "focus": {
        "color": "gray",
    },
}

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H4("MRI Regions Viewer"),
                                    ],
                                    className="header__title",
                                ),
                                html.Div(
                                    [
                                        html.P(""
                                            #"Click on the brain to add an annotation. Drag the black corners of the graph to rotate."
                                        )
                                    ],
                                    className="header__info pb-20",
                                ),
                            ],
                            className="header pb-20",
                        ),
                        html.Div(
                            id="3D-graphs",
                            children=[
                                dcc.Graph(
                                    id="brain-graph",
                                    figure={
                                        "data": create_mesh_data("human_mrt", 400),
                                        "layout": plot_layout,
                                    },
                                    config={"editable": True, "scrollZoom": False},
                                )
                            ],
                            className="graph__container",
                        ),
                    ],
                    className="container",
                )
            ],
            className="two-thirds column app__left__section",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    "Click colorscale to change", className="subheader"
                                ),
                                dcs.DashColorscales(
                                    id="colorscale-picker",
                                    colorscale=default_colorscale_index,
                                ),
                            ]
                        )
                    ],
                    className="colorscale pb-20",
                ),
                html.Div(
                    [
                        html.P("Select option", className="subheader"),
                        dcc.RadioItems(
                            #options
                            options=[
                                {"label": "Human MRT", "value": "human_mrt"},
                                {"label": "MRT Regions", "value": "mrt_regions"},
                                {"label": "Labeled Atlas", "value": "labeled_atlas"},
                                #{"label": "All Regions", "value": "all_regions"},
                            ],
                            #default
                            value="human_mrt",
                            id="radio-options",
                            labelClassName="label__option",
                            inputClassName="input__option",
                        ),
                    ],
                    className="pb-20",
                ),
                html.Div(
                    [
                        html.P("Select Slice to view", className="subheader"),
                        dcc.Slider(MIN_SLIDER_VALUE, 
                            MAX_SLIDER_VALUE,
                            value=MAX_SLIDER_VALUE,
                            id='my-slider'
                        ),
                    ],
                    className="SliceSlider pb-20",
                ),
                html.Div(
                    [
                        html.P("View Alzheimer's MRI-Slice", className="subheader"),
                        html.Hr(),
                        dcc.RadioItems(
                            options=[
                                "Non Demented",
                                "Very Mild Demented",
                                "Mild Demented",
                                "Moderate Demented",
                            ],
                            value="Non Demented",
                            id="select-alzheimer-stage",
                        ),
                        html.Br(),
                        html.Div(id='dd-output-container'),
                        html.Hr(),
                        html.Button('Show Image', id='do-show-val', n_clicks=0, style=button_style_dict),
                        html.Button('Overlay and Slice Image', id='do-slice-val', n_clicks=0, style=button_style_dict),
                        html.Button('Add Heatmap', id='add-heat-val', n_clicks=0, style=button_style_dict),
                    ],
                    className="pb-20",
                ),
                html.Div(
                    [
                        html.Div(id="output-alzheimers_image1"),
                        html.Div(id="output-alzheimers_image2"),
                    ],
                    className="pb-20",
                ),
                html.Div(
                    [
                        html.Span("Click data", className="subheader"),
                        html.Span("  |  "),
                        html.Span(
                            "Click on points in the graph.", className="small-text"
                        ),
                        dcc.Loading(
                            html.Pre(id="click-data", className="info__container"),
                            type="dot",
                        ),
                    ],
                    className="pb-20",
                ),
                html.Div(
                    [
                        html.Span("Relayout data", className="subheader"),
                        html.Span("  |  "),
                        html.Span(
                            "Drag the graph corners to rotate it.",
                            className="small-text",
                        ),
                        dcc.Loading(
                            html.Pre(id="relayout-data", className="info__container"),
                            type="dot",
                        ),
                    ],
                    className="pb-20",
                ),
            ],
            className="one-third column app__right__section",
        ),
        dcc.Store(id="annotation_storage"),
    ]
)


def add_marker(x, y, z):
    """ Create a plotly marker dict. """

    return {
        "x": [x],
        "y": [y],
        "z": [z],
        "mode": "markers",
        "marker": {"size": 25, "line": {"width": 3}},
        "name": "Marker",
        "type": "scatter3d",
        "text": ["Click point to remove annotation"],
    }


def add_annotation(x, y, z):
    """ Create plotly annotation dict. """

    return {
        "x": x,
        "y": y,
        "z": z,
        "font": {"color": "black"},
        "bgcolor": "white",
        "borderpad": 5,
        "bordercolor": "black",
        "borderwidth": 1,
        "captureevents": True,
        "ay": -100,
        "arrowcolor": "white",
        "arrowwidth": 2,
        "arrowhead": 0,
        "text": "Click here to annotate<br>(Click point to remove)",
    }


def marker_in_points(points, marker):
    """
    Checks if the marker is in the list of points.
    
    :params points: a list of dict that contains x, y, z
    :params marker: a dict that contains x, y, z
    :returns: index of the matching marker in list
    """

    for index, point in enumerate(points):
        if (
            point["x"] == marker["x"]
            and point["y"] == marker["y"]
            and point["z"] == marker["z"]
        ):
            return index
    return None

@app.callback(
    [
        Output("brain-graph", "figure"),
        Output("output-alzheimers_image2", "children"),
        Output('dd-output-container', 'children'),
        Output("output-alzheimers_image1", "children"),
    ],
    [
        #Input("brain-graph", "clickData"),
        Input("radio-options", "value"),
        Input("colorscale-picker", "colorscale"),
        Input('my-slider', 'value'),
        Input("do-slice-val", "n_clicks"),
        Input("add-heat-val", "n_clicks"),
        Input("do-show-val", "n_clicks"),
    ],
    [State("brain-graph", "clickData"), State('select-alzheimer-stage', "value"), State("brain-graph", "figure"), State("annotation_storage", "data")],
)
def brain_graph_handler(val, colorscale, z_axis, n_clicks1, n_clicks2, n_clicks3, click_data,  stage, figure, current_anno):
    """ Listener on colorscale, option picker, and graph on click to update the graph. and the slice-slider """

    graph_fig = None
    img = None
    img_out_of_labeled = None
    opacity = .6
    show_img = None

    if val== "human_mrt":
        show_stage = "Please select the option MRT Regions or Labeled Atlas to view the Alzheimer's MRI-Slice."
    else:
        show_stage = ""
    
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if (('do-slice-val' in changed_id) or ('add-heat-val' in changed_id)) and (val == "mrt_regions" ):#or val=="labeled_atlas"

        cs = [[i / (len(colorscale) - 1), rgb] for i, rgb in enumerate(colorscale)]

        if 'add-heat-val' in changed_id:
            img_with_contours, img_out_of_labeled, index, labels, heatmap = slice_img(stage, val, want_heatmap=True)
            img = heatmap
            fig = make_subplots(rows=1, cols=3, shared_yaxes=True)
            opacity=.5
        else:
            img_with_contours, img_out_of_labeled, index, labels, _ = slice_img(stage, val)
            img = img_with_contours
            fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
            opacity=.5

        fig_img = px.imshow(img_with_contours, color_continuous_scale="gray")
        fig_img.data[0]["text"] = labels
        fig_img.data[0]["hovertemplate"] = 'x: %{x}<br>y: %{y}<br>color: %{z}<br>region: %{text}<extra></extra>'
        fig.add_trace(fig_img.data[0], row=1, col=1)
        
        fig_img = px.imshow(img_out_of_labeled, color_continuous_scale=cs, zmin=-0.5e-11, zmax=1e-11)
        fig_img.data[0]["text"] = labels
        fig_img.data[0]["hovertemplate"] = 'x: %{x}<br>y: %{y}<br>color: %{z}<br>region: %{text}<extra></extra>'
        fig.add_trace(fig_img.data[0], row=1, col=2)
        
        if 'add-heat-val' in changed_id:
            heatmap = heatmap.astype("float64") * 255.
            fig_img = px.imshow(heatmap, color_continuous_scale=cs)
            fig_img.data[0]["text"] = labels
            fig_img.data[0]["hovertemplate"] = 'x: %{x}<br>y: %{y}<br>color: %{z}<br>region: %{text}<extra></extra>'
            fig.add_trace(fig_img.data[0], row=1, col=3)

        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(imshow_layout)

        if val=="mrt_regions":
            z_axis = 256-index
        else:
            z_axis = 181-index
        graph_fig = dcc.Graph(figure=fig, style={'width': 'auto', 'height': 'auto'})
        show_stage = f"You have selected {stage}"

    if "do-show-val" in changed_id:
        if stage == "Non Demented":
            show_img = np.array(Image.open("./data/images/non.jpg"))
        elif stage == "Very Mild Demented":
            show_img = np.array(Image.open("./data/images/verymild.jpg"))
        elif stage == "Mild Demented":
            show_img = np.array(Image.open("./data/images/mild.jpg"))
        else:
            show_img = np.array(Image.open("./data/images/moderate.jpg"))

        fig = px.imshow(show_img, color_continuous_scale="gray")
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(imshow_layout)

        show_img = dcc.Graph(figure=fig)

    # new option select
    if figure["data"][0]["name"] != val or z_axis != SAVED_Z:
        cs = [[i / (len(colorscale) - 1), rgb] for i, rgb in enumerate(colorscale)]
        figure["data"] = create_mesh_data(val, z_axis, img, img_out_of_labeled, cs, opacity)
        figure["layout"] = plot_layout
        for mesh in range(len(figure["data"])):
            if figure["data"][mesh]["name"] == "img":
                figure["data"][mesh]["colorscale"] = "jet"
            else:
                figure["data"][mesh]["colorscale"] = cs

        #SAVED_Z = max_z
        #MAX_SLIDER_VALUE = max_z

        return figure, graph_fig, show_stage, show_img

    # modify graph markers
    if click_data is not None and "points" in click_data:

        y_value = click_data["points"][0]["y"]
        x_value = click_data["points"][0]["x"]
        z_value = click_data["points"][0]["z"]

        #marker = add_marker(x_value, y_value, z_value)
        #point_index = marker_in_points(figure["data"], marker)
        """
        # delete graph markers
        if len(figure["data"]) > 1 and point_index is not None:
            
            figure["data"].pop(point_index)
            anno_index_offset = 2 if val == "mouse" else 1
            try:
                figure["layout"]["scene"]["annotations"].pop(
                    point_index - anno_index_offset
                )
            except Exception as error:
                print(error)
                pass
            
        # append graph markers
        else:

            # iterate through the store annotations and save it into figure data
            if current_anno is not None:
                for index, annotations in enumerate(
                    figure["layout"]["scene"]["annotations"]
                ):
                    for key in current_anno.keys():
                        if str(index) in key:
                            figure["layout"]["scene"]["annotations"][index][
                                "text"
                            ] = current_anno[key]

            figure["data"].append(marker)
            figure["layout"]["scene"]["annotations"].append(
                add_annotation(x_value, y_value, z_value)
            )
            """
            

    cs = [[i / (len(colorscale) - 1), rgb] for i, rgb in enumerate(colorscale)]
    figure["data"][0]["colorscale"] = cs

    return figure, graph_fig, show_stage, show_img


@app.callback(Output("click-data", "children"), [Input("brain-graph", "clickData")])
def display_click_data(click_data):
    return json.dumps(click_data, indent=4)


@app.callback(
    Output("relayout-data", "children"), [Input("brain-graph", "relayoutData")]
)
def display_relayout_data(relayout_data):
    return json.dumps(relayout_data, indent=4)


@app.callback(
    Output("annotation_storage", "data"),
    [Input("brain-graph", "relayoutData")],
    [State("annotation_storage", "data")],
)
def save_annotations(relayout_data, current_data):
    """ Update the annotations in the dcc store. """

    if relayout_data is None:
        raise PreventUpdate

    if current_data is None:
        return {}

    for key in relayout_data.keys():

        # to determine if the relayout has to do with annotations
        if "scene.annotations" in key:
            current_data[key] = relayout_data[key]

    return current_data


if __name__ == "__main__":
    app.run_server(debug=True)

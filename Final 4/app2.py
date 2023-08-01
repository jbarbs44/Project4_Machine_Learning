from dash import Dash, dcc, html, Input, Output, State, callback, ctx
import pandas as pd
import plotly.express as px

df = pd.read_csv('clean_data.csv')

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='Average Total Income Calculator'),
    html.H3(children=''),
    html.P(children='Enter your information to see total income'),
    html.Br(),
    dcc.Input(
        id='input-usd',
        type='number',
        placeholder="Enter the dollar amount of your loan"
    ),

    html.Button('Submit', id='btn-nclicks-1', n_clicks=0),
    html.Div(id='container-button-timestamp'),
])

@callback(
    Output('container-button-timestamp', 'children'),
    Input('btn-nclicks-1', 'n_clicks')
    )
def displayClick(btn1):
    msg = "You have not submited"
    if "btn-nclicks-1" == ctx.triggered_id:
        msg = "Submited"
    return html.Div(msg)

# app = Dash(__name__)

# app.layout = html.Div([
#     html.Div(dcc.Input(id='input-on-submit', type='text')),
#     html.Button('Submit', id='submit-val', n_clicks=0),
#     html.Div(id='container-button-basic',
#              children='Enter a value and press submit')
# ])


# @callback(
#     Output('container-button-basic', 'children'),
#     Input('submit-val', 'n_clicks'),
#     State('input-on-submit', 'value')
# )
# def update_output(n_clicks, value):
#     return 'The input value was "{}" '.format(
#         value,
#         n_clicks
#     )


if __name__ == '__main__':
    app.run(debug=True)
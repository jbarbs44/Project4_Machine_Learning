from dash import Dash, html, dcc, callback, Output, Input, State
import pandas as pd
import plotly.express as px
import pickle
import sklearn.preprocessing
import numpy as np

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)
with open("decision_tree_model.pkl","rb") as f:
    model = pickle.load(f)

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='Average Total Income Calculator'),
    html.H3(children='Enter your information to see total income'),
    html.Br(),
    html.P(children='Gender?'),
    dcc.Dropdown(
        ["M","F"],
        id='dropdown-realestate0',
        placeholder="Gender?"
    ),    
    html.P(children='Do you have a car?'),
    dcc.Dropdown(
        ["Yes","No"],
        id='dropdown-realestate1',
        placeholder="Do you have a car?"
    ),
    html.Br(),
    html.P(children='Do you own a house'),
    dcc.Dropdown(
        ["Yes","No"],
        id='dropdown-realestate2',
        placeholder="Do you own a house?"
    ),
    html.Br(),
    html.P(children='Income Category?'),
    dcc.Dropdown(
        [
            {'label': 'Working', 'value': 1},
            {'label': 'Commercial associate', 'value': 2},
            {'label': 'Pensioner', 'value': 3},
            {'label': 'State servant', 'value': 4},
            {'label': 'Student', 'value': 5}
        ],
        id='dropdown-realestate3',
        placeholder="Income Category?"
    ),
    html.Br(),
    html.P(children='Education Level?'),
    dcc.Dropdown(
        ["Higher education","Secondary / secondary special","Incomplete higher","Lower secondary","Academic degree"],
        id='dropdown-realestate4',
        placeholder="Education Level?"
    ),
    html.Br(),
    html.P(children='Marital Status?'),
    dcc.Dropdown(
        ["Single","Married","Separated","Civil Marrige","Widow"],
        id='dropdown-realestate5',
        placeholder="Marital Status?"
    ),
    html.Br(),
    html.P(children='Family Size?'),
    dcc.Input(
        id='input-family',
        type='number',
        placeholder="Family Size?"
    ),
    html.Br(),
    html.Br(),
    html.H3(children="Results"),
    html.P(children="", id="p-result"),
    html.Button('Submit', id='btn-nclicks-1', n_clicks=0),
    html.Button('Reset',id='reset_button', n_clicks=0),
])

@callback( Output('p-result', 'children'),
    Input('btn-nclicks-1', 'n_clicks'),
    State('dropdown-realestate0', 'value'),
    State('dropdown-realestate1', 'value'),
    State('dropdown-realestate2', 'value'),
    State('dropdown-realestate3', 'value'),
    State('dropdown-realestate4', 'value'),
    State('dropdown-realestate5', 'value'),
    State('input-family', 'value'),
    prevent_initial_call=True
    )
def update_result(click,gender,car, house, category, education, marital, family_size):
    if car is None:
        return "Please fill out the form"
    elif house is None:
        return "Please fill out the form"
    elif category is None:
        return "Please fill out the form"
    elif education is None:
        return "Please fill out the form"
    elif marital is None:
        return "Please fill out the form"
    elif family_size is None:
        return "Please fill out the form"
    info_for_prediction = {
        "GENDER" : 0 if gender =="F" else 1,
        "CAR" : 0 if car=='No' else 1,
        "HOUSE" : 0 if house=='No' else 1,
        "INCOME CATEGORY": int(category),
        "EDUCATION LEVEL" : str(education),
        "MARITAL STATUS": str(marital),
        "CNT_CHILDREN": str(family_size)
    }
    df_predict = pd.DataFrame(info_for_prediction,index=[0])
    df_predict = scaler.transform(df_predict)
    answer = model.predict(df_predict)
    return answer


if __name__ == '__main__':
    app.run(debug=True)
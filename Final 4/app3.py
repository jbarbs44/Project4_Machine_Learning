from dash import Dash, html, dcc, callback, Output, Input, State
import pickle
import pandas as pd

external_stylesheets = [
    'https://unpkg.com/simpledotcss/simple.min.css',
]

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)
with open("model.pkl","rb") as f:
    model = pickle.load(f)

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1(children='Loan Approvals'),
    html.H3(children='Demo for getting info from Dash'),
    html.P(children='Enter your information to see if you are approved for a loan'),
    html.Br(),
    dcc.Input(
        id='input-usd',
        type='number',
        placeholder="Enter the dollar amount of your loan"
    ),
    html.Br(),
    dcc.Input(
        id='input-term',
        type='number',
        placeholder="Enter the length of the loan in years"
    ),
    html.Br(),
    dcc.Input(
        id='input-newjobcount',
        type='number',
        placeholder="Enter the number of jobs you will create"
    ),
    html.Br(),
    dcc.Input(
        id='input-empno',
        type='number',
        placeholder="Enter the number of employees you will have"
    ),
    html.Br(),
    dcc.Dropdown(
        ["Yes","No"],
        id='dropdown-realestate',
        placeholder="Will this loan be used to purchase real estate?"
    ),
    html.Br(),
    dcc.Dropdown(
        [0,1,2],
        id='dropdown-urbanrural',
        placeholder="Select your urban-rural code"
        ),
    html.Br(),
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Br(),
    html.Br(),
    html.H3(children="Results"),
    html.P(children="", id="p-result")
])

@callback(
    Output('p-result', 'children'),
    Input('submit-val', "n_clicks"),
    State('input-usd', 'value'),
    State('input-term', 'value'),
    State('input-newjobcount', 'value'),
    State('input-empno', 'value'),
    State('dropdown-realestate', 'value'),
    State('dropdown-urbanrural', 'value'),
    prevent_initial_call=True
)
def update_result(clicks, amount_usd, term_years, 
                  count_newjobs, count_emp, realestate,
                  urbanrural):
    info_for_prediction = {
        "Amount": float(amount_usd),
        "Term": float(term_years)*12,
        "CreateJob": float(count_newjobs),
        "NoEmp": float(count_emp),
        "RealEstate": 0 if realestate=="No" else 1,
        "UrbanRural": int(urbanrural),
        }
    df_predict = pd.DataFrame(info_for_prediction,index=[0])
    df_predict = scaler.transform(df_predict)
    return model.predict(df_predict)

if __name__ == '__main__':
    app.run(debug=True)
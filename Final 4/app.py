from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px

df = pd.read_csv('Resources\Salary_Data_Based_country_and_race.csv')

#  Initialize the app
app = Dash(__name__)


app.layout = html.Div([
    html.Div(children='---'),
    dcc.Graph(figure=px.histogram(df, x='Education Level', y='Salary', histfunc='avg'))
])

# Run the app
if __name__ == '__main__':
    app.run(debug=True) 
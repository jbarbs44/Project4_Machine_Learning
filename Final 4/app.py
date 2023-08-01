from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px

df = pd.read_csv('Resource/application_record.csv')

#  Initialize the app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.Div(children='Average Income'),
    dash_table.DataTable(data=df.to_dict('records'), page_size=50),
    dcc.Graph(figure=px.histogram(df, x='AMT_INCOME_TOTAL', y='NAME_EDUCATION_TYPE', histfunc='avg'))
])


# Run the app
if __name__ == '__main__':
    app.run(debug=True) 
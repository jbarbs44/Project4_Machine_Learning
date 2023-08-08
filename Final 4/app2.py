from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import pandas as pd
import plotly.express as px
import pickle
import sklearn.preprocessing
import numpy as np
import dash_html_components as html



with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)
with open("decision_tree_model.pkl","rb") as f:
    model = pickle.load(f)

app = Dash(__name__)
app.css.append_css({"external_url": "/assets/styles.css"})

app.layout = html.Div([
    html.H1(children='Average Total Income Calculator', className="my_header"),
    html.H3(children='Enter your information to see total income'),
    html.Br(),
    html.P(children='Gender?',className="gender"),
    dcc.Dropdown(
        ["Male","Female"],
        id='dropdown-salary0',
        placeholder="Gender?"
    ),
    html.Br(),
    html.P(children='Education Level?', className="education"),
    dcc.Dropdown(
        [{'label': "High School", 'value': 1},
            {'label': "Bachelor's Degree", 'value': 2},
            {'label': "Master's Degree", 'value': 3},
            {'label': "PhD", 'value': 4}],
        id='dropdown-salary1',
        placeholder="What is your education level?"
    ),
    html.Br(),
    html.P(children='Country?', className="marital"),
    dcc.Dropdown(
        [{'label': 'UK', 'value': 1},
            {'label': 'USA', 'value': 2},
            {'label': 'Canada', 'value': 3},
            {'label': 'China', 'value': 4},
            {'label': 'Australia', 'value': 5}],
        id='dropdown-salary2',
        placeholder="What country do you live in?"
    ),
    html.Br(),
    html.P(children='Ethnicity?', className="marital"),
    dcc.Dropdown(
        [{'label': 'White', 'value': 1},
            {'label': 'Hispanic', 'value': 2},
            {'label': 'Asian', 'value': 3},
            {'label': 'African American', 'value': 4},
            {'label': 'Mixed', 'value': 5}],
        id='dropdown-salary3',
        placeholder="What is your ethnicity?"
    ),
    html.Br(),
    html.P(children='Occupation?', className="marital"),
    dcc.Dropdown(
       [{'label': 'Sales Associate', 'value': 1}, {'label': 'Project Engineer', 'value': 2}, {'label': 'Juniour HR Coordinator', 'value': 3}, {'label': 'Director of Sales', 'value': 4}, {'label': 'Director', 'value': 5}, 
        {'label': 'Social Media Man', 'value': 6}, {'label': 'IT Support Specialist', 'value': 7}, {'label': 'Customer Service Representative', 'value': 8}, {'label': 'Front End Developer', 'value': 9}, {'label': 'Help Desk Analyst', 'value': 10}, 
        {'label': 'Product Marketing Manager', 'value': 11}, {'label': 'Researcher', 'value': 12}, {'label': 'Chief Technology Officer', 'value': 13}, {'label': 'Research Director', 'value': 14}, {'label': 'Back end Developer', 'value': 15}, 
        {'label': 'Manager', 'value': 16}, {'label': 'Accountant', 'value': 17}, {'label': 'Customer Service Manager', 'value': 18}, {'label': 'Software Architect', 'value': 19}, {'label': 'Digital Content Producer', 'value': 20}, 
        {'label': 'Director of Business Development', 'value': 21}, {'label': 'Product Designer', 'value': 22}, {'label': 'Sales Executive', 'value': 23}, {'label': 'Director of Finance', 'value': 24}, {'label': 'Research Scientist', 'value': 25}, 
        {'label': 'Business Operations Analyst', 'value': 26}, {'label': 'Recruiter', 'value': 27}, {'label': 'Administrative Assistant', 'value': 28}, {'label': 'Operations Director', 'value': 29}, {'label': 'Supply Chain Analyst', 'value': 30}, 
        {'label': 'Full Stack Engineer', 'value': 31}, {'label': 'IT Support', 'value': 32}, {'label': 'Quality Assurance Analyst', 'value': 33}, {'label': 'Data Engineer', 'value': 34}, {'label': 'Chief Data Officer', 'value': 35}, 
        {'label': 'Human Resources Manager', 'value': 36}, {'label': 'Office Manager', 'value': 37}, {'label': 'Social Media Specialist', 'value': 38}, {'label': 'Developer', 'value': 39}, {'label': 'Customer Support Specialist', 'value': 40}, 
        {'label': 'Receptionist', 'value': 41}, {'label': 'Product Development Manager', 'value': 42}, {'label': 'Operations Coordinator', 'value': 43}, {'label': 'VP of Finance', 'value': 44}, {'label': 'Front end Developer', 'value': 45}, 
        {'label': 'Principal Engineer', 'value': 46}, {'label': 'Director of Marketing', 'value': 47}, {'label': 'Director of HR', 'value': 48}, {'label': 'Data Analyst', 'value': 49}, {'label': 'Marketing Analyst', 'value': 50}, 
        {'label': 'Director of Sales and Marketing', 'value': 51}, {'label': 'Marketing Director', 'value': 52}, {'label': 'Product Manager', 'value': 53}, {'label': 'Technical Recruiter', 'value': 54}, {'label': 'Engineer', 'value': 55}, 
        {'label': 'Director of Engineering', 'value': 56}, {'label': 'Sales Representative', 'value': 57}, {'label': 'Director of Product Management', 'value': 58}, {'label': 'Business Analyst', 'value': 59}, {'label': 'Strategy Consultant', 'value': 60}, 
        {'label': 'Account Manager', 'value': 61}, {'label': 'Social Media Manager', 'value': 62}, {'label': 'Supply Chain Manager', 'value': 63}, {'label': 'Customer Success Manager', 'value': 64}, {'label': 'Sales Manager', 'value': 65}, {'label': 'Scientist', 'value': 66}, 
        {'label': 'Web Developer', 'value': 67}, {'label': 'HR Generalist', 'value': 68}, {'label': 'Financial Analyst', 'value': 69}, {'label': 'Director of Human Capital', 'value': 70}, 
        {'label': 'Designer', 'value': 71}, {'label': 'Director of Human Resources', 'value': 72}, {'label': 'Marketing Specialist', 'value': 73}, {'label': 'UX Designer', 'value': 74}, {'label': 'Marketing Manager', 'value': 75}, 
        {'label': 'Copywriter', 'value': 76}, {'label': 'Customer Service Rep', 'value': 77}, {'label': 'Technical Support Specialist', 'value': 78}, {'label': 'Human Resources Director', 'value': 79}, {'label': 'Web Designer', 'value': 80}, 
        {'label': 'Data Scientist', 'value': 81}, {'label': 'Graphic Designer', 'value': 82}, {'label': 'Principal Scientist', 'value': 83}, {'label': 'Project Manager', 'value': 84}, {'label': 'Delivery Driver', 'value': 85}, 
        {'label': 'Account Executive', 'value': 86}, {'label': 'Director of Operations', 'value': 87}, {'label': 'Juniour HR Generalist', 'value': 88}, {'label': 'Content Marketing Manager', 'value': 89}, {'label': 'Sales Operations Manager', 'value': 90}, 
        {'label': 'Technical Writer', 'value': 91}, {'label': 'HR Coordinator', 'value': 92}, {'label': 'Human Resources Specialist', 'value': 93}, {'label': 'Network Engineer', 'value': 94}, {'label': 'Software Project Manager', 'value': 95}, 
        {'label': 'IT Manager', 'value': 96}, {'label': 'CEO', 'value': 97}, {'label': 'Event Coordinator', 'value': 98}, {'label': 'Software Engineer Manager', 'value': 99}, {'label': 'HR Manager', 'value': 100}, 
        {'label': 'HR Specialist', 'value': 101}, {'label': 'Business Development Associate', 'value': 102}, {'label': 'Public Relations Manager', 'value': 103}, {'label': 'Operations Analyst', 'value': 104}, {'label': 'Director of Data Science', 'value': 105}, 
        {'label': 'Marketing Coordinator', 'value': 106}, {'label': 'Training Specialist', 'value': 107}, {'label': 'IT Project Manager', 'value': 108}, {'label': 'Financial Manager', 'value': 109}, {'label': 'Operations Manager', 'value': 110}, 
        {'label': 'Software Manager', 'value': 111}, {'label': 'Software Developer', 'value': 112}, {'label': 'Creative Director', 'value': 113}, {'label': 'Project Coordinator', 'value': 114}, {'label': 'Human Resources Coordinator', 'value': 115}, 
        {'label': 'Software Engineer', 'value': 116}, {'label': 'Business Intelligence Analyst', 'value': 117}, {'label': 'Business Development Manager', 'value': 118}, {'label': 'Customer Success Rep', 'value': 119}, {'label': 'Sales Director', 'value': 120}, 
        {'label': 'UX Researcher', 'value': 121}, {'label': 'IT Consultant', 'value': 122}, {'label': 'Digital Marketing Specialist', 'value': 123}, {'label': 'Consultant', 'value': 124}, {'label': 'Data Entry Clerk', 'value': 125}, 
        {'label': 'Advertising Coordinator', 'value': 126}, {'label': 'VP of Operations', 'value': 127}, {'label': 'Financial Advisor', 'value': 128}, {'label': 'Digital Marketing Manager', 'value': 129}],
        id='dropdown-salary4',
        placeholder="What is your current Occupation?"
    ),
    html.Br(),
    html.P(children='Age?', className="family"),
    dcc.Input(
        id='input-age',
        type='number',
        placeholder="What is your age?"
    ),
    html.Br(),
    html.P(children='Years of Experience?', className="family"),
    dcc.Input(
        id='input-years',
        type='number',
        placeholder="How many years of experience do you have?"
    ),
    html.Br(),
    html.Br(),
    html.H3(children="Results"),
    html.P(children="", id="p-result"),
    html.Button('Submit', id='btn-nclicks-1', n_clicks=0, className='summit'),
    html.Button('Reset',id='reset_button', n_clicks=0,className='reset'),
])

@callback( Output('p-result', 'children'),
    Input('btn-nclicks-1', 'n_clicks'),
    State('dropdown-salary0', 'value'),
    State('dropdown-salary1', 'value'),
    State('dropdown-salary2', 'value'),
    State('dropdown-salary3', 'value'),
    State('dropdown-salary4', 'value'),
    State('input-age', 'value'),
    State('input-years', 'value'),
    prevent_initial_call=True,
    )
def update_result(click,gender,education, country, ethnicity, occupation, age, years):
    if gender is None:
        return "Please fill out the gender section."
    elif education is None:
        return "Please fill out education section."
    elif country is None:
        return "Please fill out country section."
    elif education is None:
        return "Please fill out the education level section."
    elif ethnicity is None:
        return "Please fill out the ethnicity section."
    elif occupation is None:
        return "Please fill out the occupation section."
    elif age is None:
        return "Please fill out the age section."
    elif years is None:
        return "Please fill out the years of experience section."
    info_for_prediction = {
        "Age": float(age),
        "Gender" : 0 if gender =="Female" else 1,
        "Education": int(education),
        "Job": int(occupation),
        "Years of Experience": float(years),
        "Country": int(country),
        "Race" : int(ethnicity),

    }
    df_predict = pd.DataFrame(info_for_prediction,index=[0])
    df_predict = scaler.transform(df_predict)
    answer = model.predict(df_predict)
    html.Br(),
    formatted_answer = '${:,.2f}'.format(answer.item())
    return f' You make {formatted_answer} per year'

@callback(
    Output('dropdown-salary0', 'value'),
    Output('dropdown-salary1', 'value'),
    Output('dropdown-salary2', 'value'),
    Output('dropdown-salary3', 'value'),
    Output('dropdown-salary4', 'value'),
    Output('input-age', 'value'),
    Output('input-years', 'value'),
    Output('p-result', 'children',allow_duplicate=True),
    Input('reset_button', "n_clicks"),
    prevent_initial_call='initial_duplicate'
)
def reset(clicks):
    return f'',None,None,None,None,None,None,None


if __name__ == '__main__':
    app.run(debug=True)
from dash import Dash, html, dcc, callback, Output, Input, State
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
        [{'label': "Bachelor's", 'value': 1},
            {'label': "Master's", 'value': 2},
            {'label': 'PhD', 'value': 3},
            {'label': "Bachelor's Degree", 'value': 4},
            {'label': "Master's Degree", 'value': 5},
            {'label': 'High School', 'value': 6},],
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
            {'label': 'Mixed', 'value': 5},],
        id='dropdown-salary3',
        placeholder="What is your ethnicity?"
    ),
    html.Br(),
    html.P(children='Occupation?', className="marital"),
    dcc.Dropdown(
          [{'label': 'Software Engineer', 'value': 1}, {'label': 'Data Analyst', 'value': 2}, {'label': ' Manager', 'value': 3}, {'label': 'Sales Associate', 'value': 4}, {'label': 'Director', 'value': 5},
            {'label': 'Marketing Analyst', 'value': 6}, {'label': 'Product Manager', 'value': 7}, {'label': 'Sales Manager', 'value': 8}, {'label': 'Marketing Coordinator', 'value': 9}, {'label': ' Scientist', 'value': 10},
            {'label': 'Software Developer', 'value': 11}, {'label': 'HR Manager', 'value': 12}, {'label': 'Financial Analyst', 'value': 13}, {'label': 'Project Manager', 'value': 14}, {'label': 'Customer Service Rep', 'value': 15},
            {'label': 'Operations Manager', 'value': 16}, {'label': 'Marketing Manager', 'value': 17}, {'label': ' Engineer', 'value': 18}, {'label': 'Data Entry Clerk', 'value': 19}, {'label': 'Sales Director', 'value': 20}, 
            {'label': 'Business Analyst', 'value': 21}, {'label': 'VP of Operations', 'value': 22}, {'label': 'IT Support', 'value': 23}, {'label': 'Recruiter', 'value': 24}, {'label': 'Financial Manager', 'value': 25}, 
            {'label': 'Social Media Specialist', 'value': 26}, {'label': 'Software Manager', 'value': 27}, {'label': ' Developer', 'value': 28}, {'label': ' Consultant', 'value': 29}, {'label': 'Product Designer', 'value': 30}, 
            {'label': 'CEO', 'value': 31}, {'label': 'Accountant', 'value': 32}, {'label': 'Data Scientist', 'value': 33}, {'label': 'Marketing Specialist', 'value': 34}, {'label': 'Technical Writer', 'value': 35}, 
            {'label': 'HR Generalist', 'value': 36}, {'label': 'Project Engineer', 'value': 37}, {'label': 'Customer Success Rep', 'value': 38}, {'label': 'Sales Executive', 'value': 39}, {'label': 'UX Designer', 'value': 40}, 
            {'label': 'Operations Director', 'value': 41}, {'label': 'Network Engineer', 'value': 42}, {'label': 'Administrative Assistant', 'value': 43}, {'label': 'Strategy Consultant', 'value': 44}, {'label': 'Copywriter', 'value': 45}, 
            {'label': 'Account Manager', 'value': 46}, {'label': 'Director of Marketing', 'value': 47}, {'label': 'Help Desk Analyst', 'value': 48}, {'label': 'Customer Service Manager', 'value': 49}, {'label': 'Business Intelligence Analyst', 'value': 50}, 
            {'label': 'Event Coordinator', 'value': 51}, {'label': 'VP of Finance', 'value': 52}, {'label': 'Graphic Designer', 'value': 53}, {'label': 'UX Researcher', 'value': 54}, {'label': 'Social Media Manager', 'value': 55}, 
            {'label': 'Director of Operations', 'value': 56}, {'label': ' Data Scientist', 'value': 57}, {'label': ' Accountant', 'value': 58}, {'label': 'Digital Marketing Manager', 'value': 59}, {'label': 'IT Manager', 'value': 60}, 
            {'label': 'Customer Service Representative', 'value': 61}, {'label': 'Business Development Manager', 'value': 62}, {'label': ' Financial Analyst', 'value': 63}, {'label': 'Web Developer', 'value': 64}, {'label': 'Research Director', 'value': 65}, 
            {'label': 'Technical Support Specialist', 'value': 66}, {'label': 'Creative Director', 'value': 67}, {'label': ' Software Engineer', 'value': 68}, {'label': 'Human Resources Director', 'value': 69}, {'label': 'Content Marketing Manager', 'value': 70}, 
            {'label': 'Technical Recruiter', 'value': 71}, {'label': 'Sales Representative', 'value': 72}, {'label': 'Chief Technology Officer', 'value': 73}, {'label': ' Designer', 'value': 74}, {'label': 'Financial Advisor', 'value': 75}, 
            {'label': ' Account Manager', 'value': 76}, {'label': ' Project Manager', 'value': 77}, {'label': 'Principal Scientist', 'value': 78}, {'label': 'Supply Chain Manager', 'value': 79}, {'label': ' Marketing Manager', 'value': 80}, 
            {'label': 'Training Specialist', 'value': 81}, {'label': 'Research Scientist', 'value': 82}, {'label': ' Software Developer', 'value': 83}, {'label': 'Public Relations Manager', 'value': 84}, {'label': 'Operations Analyst', 'value': 85}, 
            {'label': 'Product Marketing Manager', 'value': 86}, {'label': ' HR Manager', 'value': 87}, {'label': ' Web Developer', 'value': 88}, {'label': ' Project Coordinator', 'value': 89}, {'label': 'Chief Data Officer', 'value': 90}, 
            {'label': 'Digital Content Producer', 'value': 91}, {'label': 'IT Support Specialist', 'value': 92}, {'label': ' Marketing Analyst', 'value': 93}, {'label': 'Customer Success Manager', 'value': 94}, {'label': ' Graphic Designer', 'value': 95}, 
            {'label': 'Software Project Manager', 'value': 96}, {'label': 'Supply Chain Analyst', 'value': 97}, {'label': ' Business Analyst', 'value': 98}, {'label': 'Office Manager', 'value': 99}, {'label': 'Principal Engineer', 'value': 100}, 
            {'label': ' HR Generalist', 'value': 101}, {'label': ' Product Manager', 'value': 102}, {'label': ' Operations Analyst', 'value': 103}, {'label': 'Sales Operations Manager', 'value': 104}, {'label': ' Web Designer', 'value': 105}, 
            {'label': ' Training Specialist', 'value': 106}, {'label': ' Research Scientist', 'value': 107}, {'label': ' Sales Representative', 'value': 108}, {'label': ' Data Analyst', 'value': 109}, {'label': ' Product Marketing Manager', 'value': 110}, 
            {'label': ' Sales Manager', 'value': 111}, {'label': ' Marketing Specialist', 'value': 112}, {'label': 'Director of Sales', 'value': 113}, {'label': ' Recruiter', 'value': 114}, {'label': ' Business Development Manager', 'value': 115}, 
            {'label': ' Product Designer', 'value': 116}, {'label': ' Customer Support Specialist', 'value': 117}, {'label': ' IT Support Specialist', 'value': 118}, {'label': ' Operations Manager', 'value': 119}, {'label': 'Director of Human Resources', 'value': 120}, 
            {'label': 'Director of Product Management', 'value': 121}, {'label': ' Copywriter', 'value': 122}, {'label': ' Marketing Coordinator', 'value': 123}, {'label': ' Human Resources Manager', 'value': 124}, {'label': ' Business Development Associate', 'value': 125}, 
            {'label': ' Researcher', 'value': 126}, {'label': ' HR Coordinator', 'value': 127}, {'label': 'Director of Finance', 'value': 128}, {'label': ' Human Resources Coordinator', 'value': 129}, {'label': ' UX Designer', 'value': 130}, 
            {'label': ' IT Project Manager', 'value': 131}, {'label': ' Quality Assurance Analyst', 'value': 132}, {'label': 'Director of Sales and Marketing', 'value': 133}, {'label': ' Account Executive', 'value': 134}, {'label': 'Director of Business Development', 'value': 135}, 
            {'label': ' Social Media Manager', 'value': 136}, {'label': ' Human Resources Specialist', 'value': 137}, {'label': 'Director of Human Capital', 'value': 138}, {'label': ' Advertising Coordinator', 'value': 139}, {'label': ' Marketing Director', 'value': 140}, 
            {'label': ' IT Consultant', 'value': 141}, {'label': ' Financial Advisor', 'value': 142}, {'label': ' Business Operations Analyst', 'value': 143}, {'label': ' Social Media Specialist', 'value': 144}, {'label': ' Product Development Manager', 'value': 145}, 
            {'label': ' Software Architect', 'value': 146}, {'label': ' Financial Manager', 'value': 147}, {'label': ' HR Specialist', 'value': 148}, {'label': ' Data Engineer', 'value': 149}, {'label': ' Operations Coordinator', 'value': 150}, 
            {'label': 'Director of HR', 'value': 151}, {'label': 'Director of Engineering', 'value': 152}, {'label': 'Software Engineer Manager', 'value': 153}, {'label': 'Back end Developer', 'value': 154}, {'label': ' Project Engineer', 'value': 155}, 
            {'label': 'Full Stack Engineer', 'value': 156}, {'label': 'Front end Developer', 'value': 157}, {'label': 'Front End Developer', 'value': 158}, {'label': 'Director of Data Science', 'value': 159}, {'label': 'Human Resources Coordinator', 'value': 160}, 
            {'label': ' Sales Associate', 'value': 161}, {'label': 'Human Resources Manager', 'value': 162}, {'label': 'Juniour HR Generalist', 'value': 163}, {'label': 'Juniour HR Coordinator', 'value': 164}, {'label': 'Digital Marketing Specialist', 'value': 165}, 
            {'label': 'Receptionist', 'value': 166}, {'label': 'Marketing Director', 'value': 167}, {'label': 'Social Media Man', 'value': 168}, {'label': 'Delivery Driver', 'value': 169}],
        id='dropdown-salary4',
        placeholder="What is your ethnicity?"
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
    html.Button('Submit', id='btn-nclicks-1', n_clicks=0),
    html.Button('Reset',id='reset_button', n_clicks=0),
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
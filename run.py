from module import *
from predictions import *

page = st.sidebar.radio('Go to', ['Home', 'EDA', 'Make Predictions'])

if page=='Home':
    st.sidebar.title('navigation')
    add_bg_image("images/auto+insurance.webp")
    st.markdown("""
    <h1 style='text-align: center;'>Health Insurance Cross-Sell Prediction</h1>""",unsafe_allow_html=True)
    st.subheader("Problem Statement")
    st.write("""
    An insurance company that provides health insurance to its customers wants to know 
    whether these existing customers would also be interested in purchasing vehicle 
    insurance from the same company. Identifying likely responders in advance allows 
    the company to plan a more effective and targeted cross-sell marketing campaign, 
    rather than reaching out to its entire customer base indiscriminately.

    The dataset contains demographic details (age, gender, region), vehicle information 
    (age, past damage), and existing policy details (previously insured status, annual 
    premium, sales channel, vintage) for over 380,000 customers, along with whether 
    each customer responded positively to a similar offer in the past.
    """)

    st.subheader("Objective")
    st.write("""
    To build a binary classification model using an Artificial Neural Network (ANN) 
    that predicts whether a customer is likely to be interested in availing vehicle 
    insurance, based on their demographic and policy-related information.

    The key goals of this project are to:
    - Perform in-depth exploratory data analysis to identify the customer attributes 
    most strongly associated with cross-sell interest.
    - Handle the significant class imbalance present in the target variable using 
    appropriate techniques (class weighting).
    - Build, train, and tune an ANN to maximize the model's ability to correctly 
    identify genuinely interested customers (recall), which is more valuable than 
    raw accuracy in a targeted marketing context.
    - Deploy the trained model through an interactive application that allows 
    real-time prediction on new customer data.
    """)
elif page=='EDA':
    add_bg_image("images/eda.jpg")
    eda()
else:
    add_bg_image("images/got_insurance.jpg")
    predictions()
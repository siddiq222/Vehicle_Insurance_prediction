#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import base64
import io
import warnings
warnings.filterwarnings('ignore')
import streamlit as st



cat_cols=['Gender','Driving_License','Previously_Insured','Vehicle_Age','Vehicle_Damage','Response','Region_Code','Policy_Sales_Channel']
num_cols=['Age','Annual_Premium','Vintage']
b_c=['Shape','Head','Tail','Info','Describe']


with open("insights.json", "r") as f:
    data = json.load(f)


df=pd.read_csv(r'train.csv')
sample=df.sample(n=50000,random_state=42)


def add_bg_image(image_file):
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            /* Full screen background */
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}

            /* Dark overlay for readability */
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0; left: 0;
                width: 100%; height: 100%;
                background: rgba(0, 0, 0, 0.55);
                z-index: 0;
            }}

            /* Keep content above overlay */
            .stApp > * {{
                position: relative;
                z-index: 1;
            }}

            /* Make main block transparent */
            .block-container {{
                background: transparent !important;
            }}

            /* Sidebar background */
            section[data-testid="stSidebar"] {{
                background: rgba(0, 0, 0, 0.6) !important;
            }}

            /* White text for visibility */
            h1, h2, h3, p, label, .stMarkdown {{
                color: white !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )


def eda():
    st.markdown("""
            <style>
            [data-testid="stSelectbox"] select,
            [data-testid="stSelectbox"] > div > div {
                background-color: rgba(255, 255, 255, 0.15) !important;
                color: white !important;
                border: 1px solid rgba(255, 255, 255, 0.4) !important;
                border-radius: 6px !important;
            }
            
            /* ── Number Input ── */
            [data-testid="stNumberInput"] input {
                background-color: rgba(255, 255, 255, 0.15) !important;
                color: white !important;
                border: 1px solid rgba(255, 255, 255, 0.4) !important;
                border-radius: 6px !important;
            }

            /* +/- buttons */
            [data-testid="stNumberInput"] button {
               background-color: rgba(255, 255, 255, 0.15) !important;
               color: white !important;
               border: 1px solid rgba(255, 255, 255, 0.4) !important;
            }

            [data-testid="stNumberInput"] button:hover {
                background-color: #f0f0f0 !important;
            }
            </style>
        """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    /* Broader selector for dropdown portal */
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background-color: rgba(255, 255, 255, 0.15) !important;
        color: white !important;
    }

    ul[data-testid="stSelectboxVirtualDropdown"] li {
        background-color: rgba(255, 255, 255, 0.15) !important;
        color: white !important;
    }
                
    ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
        background-color: #f0f0f0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    def basic_checks(df):
        check=st.radio('pick one',b_c,horizontal=True)
        if check=='Shape':
            rows, cols = df.shape
            st.metric(label="Dataset Shape", value=f"{rows:,} rows × {cols} cols")
        elif check=='Head':
            html_table = df.head(10).to_html()
            st.markdown(f"""
                <div style="
                    overflow-x: auto;
                    overflow-y: auto;
                    max-height: 350px;
                    background: rgba(255,255,255,0.1);
                    border-radius: 8px;
                    padding: 10px;
                ">
                    <style>
                        .custom-table {{ border-collapse: collapse; width: 100%; }}
                        .custom-table th, .custom-table td {{
                            padding: 8px 14px;
                            color: white;
                            border: 1px solid rgba(255,255,255,0.2);
                            white-space: nowrap;
                        }}
                        .custom-table th {{ background: rgba(255,255,255,0.15); }}
                        .custom-table tr:hover td {{ background: rgba(255,255,255,0.1); }}
                    </style>
                    {html_table.replace('<table', '<table class="custom-table"')}
                </div>
            """, unsafe_allow_html=True)
        elif check=='Tail':
            html_table = df.tail(10).to_html()
            st.markdown(f"""
                <div style="
                    overflow-x: auto;
                    overflow-y: auto;
                    max-height: 350px;
                    background: rgba(255,255,255,0.1);
                    border-radius: 8px;
                    padding: 10px;
                ">
                    <style>
                        .custom-table {{ border-collapse: collapse; width: 100%; }}
                        .custom-table th, .custom-table td {{
                            padding: 8px 14px;
                            color: white;
                            border: 1px solid rgba(255,255,255,0.2);
                            white-space: nowrap;
                        }}
                        .custom-table th {{ background: rgba(255,255,255,0.15); }}
                        .custom-table tr:hover td {{ background: rgba(255,255,255,0.1); }}
                    </style>
                    {html_table.replace('<table', '<table class="custom-table"')}
                </div>
            """, unsafe_allow_html=True)
        elif check=='Info':
            info_df=pd.DataFrame({"Column": df.columns,
                        "Non-Null Count": df.notnull().sum().values,
                        "Null Count": df.isnull().sum().values,
                        "Dtype": df.dtypes.values,
                        "Unique Values": df.nunique().values})
            st.table(info_df)
        else:
            nc=st.selectbox('pick data type',['numerical','categorical'])
            if nc=='numerical':
                html_table = df.describe().to_html()
                st.markdown(f"""
                    <div style="
                        overflow-x: auto;
                        overflow-y: auto;
                        max-height: 350px;
                        background: rgba(255,255,255,0.1);
                        border-radius: 8px;
                        padding: 10px;
                    ">
                        <style>
                            .custom-table {{ border-collapse: collapse; width: 100%; }}
                            .custom-table th, .custom-table td {{
                                padding: 8px 14px;
                                color: white;
                                border: 1px solid rgba(255,255,255,0.2);
                                white-space: nowrap;
                            }}
                            .custom-table th {{ background: rgba(255,255,255,0.15); }}
                            .custom-table tr:hover td {{ background: rgba(255,255,255,0.1); }}
                        </style>
                        {html_table.replace('<table', '<table class="custom-table"')}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.table(df.describe(include='object'))
    def univariate():
        dtype=st.selectbox('select data type',['categorical','numerical'])
        if dtype=='categorical':
            columns=st.selectbox('select a column',cat_cols)
            if columns=='Region_Code' or 'Policy_Sales_Channel':
                fig,ax=plt.subplots(figsize=(4,4))
                sns.countplot(x=sample[columns],ax=ax,order=sample[columns].value_counts().iloc[:10].index)
                st.pyplot(fig)
            else:
                fig,ax=plt.subplots(figsize=(4,4))
                sns.countplot(x=sample[columns],ax=ax)
                st.pyplot(fig)
        else:
            columns=st.selectbox('select a column',num_cols)
            if columns:
                fig,ax=plt.subplots(figsize=(4,4))
                sns.boxplot(sample[columns],ax=ax)
                st.pyplot(fig)
        ins=st.button('Insight')
        if ins:
            st.subheader(data['univariate'])
    def bivariate():
        b_cat=[]
        b_num=[]
        type_anal=st.selectbox('pick one',['Numerical vs Target','Categorical vs Target'])
        for col in cat_cols:
            b_cat.append(col+' vs Response')
        for col in num_cols:
            b_num.append(col+' vs Response')
        if type_anal=='Numerical vs Target':
            anal=st.selectbox('select columns',b_num)
            if anal:
                c1=anal.split()[0]
                fig,ax=plt.subplots(figsize=(4,4))
                sns.boxplot(x='Response', y=c1, data=sample,ax=ax)
                st.pyplot(fig)
        else:
            anal=st.selectbox('select columns',b_cat)
            c1=anal.split()[0]
            if c1=='Region_Code' or 'Policy_Sales_Channel':
                top10 = sample[col].value_counts().iloc[:10].index
                subset = sample[sample[col].isin(top10)]
                ct = pd.crosstab(subset[col], subset['Response'], normalize='index') * 100
                ct = ct.loc[top10]
                fig,ax=plt.subplots(figsize=(4,4))
                ct.plot(kind='bar', stacked=True,ax=ax)
                plt.ylabel('Percentage %')
                plt.legend(title='Response', labels=['No (0)', 'Yes (1)'])
                st.pyplot(fig)
            else:
                fig,ax=plt.subplots(figsize=(4,4))
                ct = pd.crosstab(sample[anal.split()[0]], sample['Response'], normalize='index') * 100
                ct.plot(kind='bar', stacked=True,ax=ax)
                plt.ylabel('Percentage %')
                plt.legend(title='Response', labels=['No (0)', 'Yes (1)'])
                st.pyplot(fig)
        ins=st.button('Insight')
        if ins:
            st.subheader(data['bivariate'])


    def multivariate():
        l1=['All','Vehicle_Damage vs Previously_Insured vs Response','Vehicle_Age vs Age vs Response']
        ch=st.selectbox('pick one',l1)
        if ch=='All':
            fig,ax=plt.subplots(figsize=(4,4))
            sns.heatmap(data=sample[num_cols].corr(),annot=True,vmin=-1,vmax=1,linewidths=1,ax=ax)
            st.pyplot(fig)
        else:
            c21=ch.split()[0]
            c23=ch.split()[2]
            if c21=='Vehicle_Damage':
                fig=sns.catplot(x=c21,hue=c23,col='Response',kind='count',data=sample)
                st.pyplot(fig)
            else:
                fig,ax=plt.subplots(figsize=(4,4))
                sns.boxplot(x=c21, y=c23, hue='Response', data=sample,ax=ax)
                st.pyplot(fig)
        ins=st.button('Insight')
        if ins:
            st.subheader(data['multivariate'])
    
    t1=st.selectbox('choose one',['Basic Checks','Univariate Analysis','Bivariate Analysis','Multivariate Analysis'])
    if t1=='Basic Checks':
        var=st.selectbox('pick one',['Original Dataset','Sampled Dataset'])
        if var=='Original Dataset':
            basic_checks(df)
        else:
            basic_checks(sample)
    elif t1=='Univariate Analysis':
        univariate()
    elif t1=='Bivariate Analysis':
        bivariate()
    else:
        multivariate()
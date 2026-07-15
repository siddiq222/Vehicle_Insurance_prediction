from module import *
import joblib
from tensorflow.keras.models import load_model


model = load_model('final_ann_model.h5')
scaler = joblib.load('Preprocessing_Techniques\scaler.pkl')
region_freq = joblib.load('Preprocessing_Techniques\\region_freq.pkl')
channel_freq = joblib.load('Preprocessing_Techniques\channel_freq.pkl')
ohe = joblib.load('Preprocessing_Techniques\onehot_encoder.pkl')
vehicle_age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
ohe_cols = ['Gender', 'Vehicle_Damage']
num_cols = ['Age', 'Annual_Premium', 'Vintage', 'Region_Code_freq', 'Policy_Sales_Channel_freq']
FINAL_COLUMN_ORDER = ['Age','Driving_License','Previously_Insured','Vehicle_Age','Annual_Premium','Vintage','Gender_Male','Vehicle_Damage_Yes','Region_Code_freq','Policy_Sales_Channel_freq']

def predictions():
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
                background-color: rgba(14, 97, 78, 0.3) !important;
                color: white !important;
                border: 1px solid rgba(255, 255, 255, 0.15) !important;
                border-radius: 6px !important;
            }

            /* +/- buttons */
            [data-testid="stNumberInput"] button {
                background-color: rgba(14, 97, 78, 0.3) !important;
                color: white !important;
                border: 1px solid rgba(255, 255, 255, 0.15) !important;
            }

            [data-testid="stNumberInput"] button:hover {
                background-color: rgba(14, 97, 78, 0.5) !important;
                box-shadow: none !important;
            }
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

    
    gender=st.selectbox('Gender',['Male','Female'])
    age=st.number_input('Age',min_value=0,max_value=100)
    driving_lisence=st.radio('Driving_lisence',['Yes','No'])
    Region_code=st.number_input('Region code')
    previously_insured=st.radio('Previously_Insured',['Yes','No'])
    vehicle_age=st.selectbox('Vehicle Age',['< 1 Year','1-2 Year','> 2 Years'])
    vehicle_damage=st.radio('Vehicle Damage',['Yes','No'])
    annual_premium=st.number_input('Annual Premium')
    policy_sales_channel=st.number_input('policy sales channel')
    vintage=st.number_input('Vintage')
    new_customer_df = pd.DataFrame([{
    'Gender': gender,
    'Age': age,
    'Driving_License': 1 if driving_lisence=='Yes' else 0,
    'Region_Code': Region_code,
    'Previously_Insured': 1 if previously_insured=='Yes' else 0,
    'Vehicle_Age': vehicle_age,
    'Vehicle_Damage': vehicle_damage,
    'Annual_Premium': annual_premium,
    'Policy_Sales_Channel': policy_sales_channel,
    'Vintage': vintage
    }])
    def preprocess(rawdf):
        df=rawdf.copy()

        df['Vehicle_Age'] = df['Vehicle_Age'].map(vehicle_age_map)

        #feature engineering
        df['Region_Code_freq'] = df['Region_Code'].map(region_freq).fillna(0)
        df['Policy_Sales_Channel_freq'] = df['Policy_Sales_Channel'].map(channel_freq).fillna(0)
        df = df.drop(columns=['Region_Code', 'Policy_Sales_Channel'])

        #encoding nominal features
        encoded = pd.DataFrame(
            ohe.transform(df[ohe_cols]),
            columns=ohe.get_feature_names_out(ohe_cols),
            index=df.index
        )
        df = pd.concat([df.drop(columns=ohe_cols), encoded], axis=1)

        # Log transform Annual_Premium
        df['Annual_Premium'] = np.log1p(df['Annual_Premium'])

        # Scaling numerical columns
        df[num_cols] = scaler.transform(df[num_cols])
        df = df[FINAL_COLUMN_ORDER]

        return df

    b = st.button('Predict')
    if b:
        processed = preprocess(new_customer_df)
        prob = model.predict(processed)
        prob_value = prob[0][0]
                
        if prob_value > 0.5:
            st.success(f"Customer is Interested in Availing Insurance (Confidence: {prob_value:.2%})")
        else:
            st.warning(f"Customer is Not Interested in Availing Insurance (Confidence: {(1 - prob_value):.2%})")
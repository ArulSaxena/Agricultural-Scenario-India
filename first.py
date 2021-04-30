import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import style


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

st.set_page_config(page_title='Agricultural Scenario', page_icon='🌿',layout="centered")
# Dataset import
df=pd.read_csv('Data/streamlit_data.csv')
crop=pd.read_csv("Data/District-wise, season-wise crop production statistics.csv")
faostat=pd.read_csv("Data/FAOSTAT_data_2-27-2021.csv")
rain_df=pd.read_csv("Data/All India area weighted monthly, seasonal and annual rainfall (in mm) from 1901-2015.csv")
temp_df=pd.read_csv("Data/Mean_Temperatures_India1901-2012.csv")

state_code={"State_Name_0":1,"State_Name_1":2,"State_Name_2":3,"State_Name_3":4,"State_Name_4":5,"State_Name_5":6,"State_Name_6":7,
"State_Name_7":8,"State_Name_8":9,"State_Name_9":10,"State_Name_10":11,"State_Name_11":12,"State_Name_12":13,"State_Name_13":14,
"State_Name_14":15,"State_Name_15":16,"State_Name_16":17,"State_Name_17":18,"State_Name_18":19,"State_Name_19":20,"State_Name_20":21,
"State_Name_21":22,"State_Name_22":23,"State_Name_23":24,"State_Name_24":25,"State_Name_25":26,"State_Name_26":27,"State_Name_27":28,
"State_Name_28":29,"State_Name_29":30,"State_Name_30":31}

State_id={'Andhra Pradesh': 2,
 'Arunachal Pradesh': 3,
 'Assam': 4,
 'Bihar': 5,
 'Chandigarh': 6,
 'Chhattisgarh': 7,
 'Dadra and Nagar Haveli': 8,
 'Daman and Diu': 9,
 'Delhi': 10,
 'Goa': 11,
 'Gujarat': 12,
 'Haryana': 13,
 'Himachal Pradesh': 14,
 'Jharkhand': 16,
 'Karnataka': 17,
 'Kerala': 18,
 'Lakshadweep': 19,
 'Madhya Pradesh': 20,
 'Maharashtra': 21,
 'Manipur': 22,
 'Meghalaya': 23,
 'Mizoram': 24,
 'Nagaland': 25,
 'Puducherry': 27,
 'Punjab': 28,
 'Rajasthan': 29,
 'Sikkim': 30,
 'Tamil Nadu': 31,
 'Tripura': 32,
 'Uttar Pradesh': 33,
 'West Bengal': 35,
 'Andaman and Nicobar Islands': 1,
 'Jammu and Kashmir ': 15,
 'Odisha': 26,
 'Uttarakhand': 34}

crop_id={'Arecanut': 1,
 'Other Kharif pulses': 2,
 'Rice': 3,
 'Banana': 4,
 'Cashewnut': 5,
 'Coconut ': 6,
 'Dry ginger': 7,
 'Sugarcane': 8,
 'Sweet potato': 9,
 'Tapioca': 10,
 'other oilseeds': 11,
 'Arhar/Tur': 12,
 'Bajra': 13,
 'Castor seed': 14,
 'Cotton(lint)': 15,
 'Dry chillies': 16,
 'Gram': 17,
 'Groundnut': 18,
 'Horse-gram': 19,
 'Jowar': 20,
 'Korra': 21,
 'Maize': 22,
 'Moong(Green Gram)': 23,
 'Onion': 24,
 'other misc. pulses': 25,
 'Ragi': 26,
 'Samai': 27,
 'Sesamum': 28,
 'Small millets': 29,
 'Sunflower': 30,
 'Urad': 31,
 'Linseed': 32,
 'Safflower': 33,
 'Wheat': 34,
 'Coriander': 35,
 'Potato': 36,
 'Tobacco': 37,
 'Turmeric': 38,
 'Mesta': 39,
 'Other  Rabi pulses': 40,
 'Rapeseed &Mustard': 41,
 'Niger seed': 42,
 'Varagu': 43,
 'Oilseeds total': 44,
 'Pulses total': 45,
 'Jute': 46,
 'Barley': 47,
 'Khesari': 48,
 'Masoor': 49,
 'Peas & beans (Pulses)': 50,
 'Garlic': 51,
 'Soyabean': 52,
 'Sannhamp': 53,
 'Moth': 54,
 'Guar seed': 55,
 'Other Cereals & Millets': 56,
 'Black pepper': 57,
 'Cardamom': 58,
 'Kapas': 59,
 'Tea': 60,
 'Jute & mesta': 61,
 'Rubber': 62,
 'Coffee': 63,
 'Beans & Mutter(Vegetable)': 64,
 'Bhindi': 65,
 'Brinjal': 66,
 'Citrus Fruit': 67,
 'Cucumber': 68,
 'Grapes': 69,
 'Mango': 70,
 'Orange': 71,
 'other fibres': 72,
 'Other Fresh Fruits': 73,'Other Vegetables': 74, 'Papaya': 75,
 'Pome Fruit': 76, 'Tomato': 77,
 'Cabbage': 78, 'Peas  (vegetable)': 79,
 'Bottle Gourd': 80, 'Pineapple': 81,
 'Turnip': 82, 'Carrot': 83,
 'Redish': 84, 'Arcanut (Processed)': 85,
 'Atcanut (Raw)': 86, 'Cashewnut Processed': 87,
 'Cashewnut Raw': 88, 'Bitter Gourd': 89,
 'Drum Stick': 90, 'Jack Fruit': 91,
 'Snak Guard': 92,'Cauliflower': 93,
 'Water Melon': 94, 'Ash Gourd': 95,
 'Beet Root': 96,
 'Lab-Lab': 97,
 'Other Citrus Fruit': 98,
 'Pome Granet': 99,
 'Ribed Guard': 100,
 'Yam': 101,
 'Pump Kin': 102,
 'Apple': 103,
 'Peach': 104,
 'Pear': 105,
 'Plums': 106,
 'Ber': 107,
 'Litchi': 108,
 'Ginger': 109,
 'Cowpea(Lobia)': 110,
 'Paddy': 111,
 'Total foodgrain': 112,
 'Blackgram': 113,
 'Cond-spcs other': 114,
 'Lemon': 115,
 'Sapota': 116}


crop_code={"Crop_0":1,"Crop_1":2,"Crop_2":3,"Crop_3":4,"Crop_4":5,"Crop_5":6,"Crop_6":7,"Crop_7":8,"Crop_8":9,
      "Crop_9":10,"Crop_10":11,"Crop_11":12,"Crop_12":13,"Crop_13":14,"Crop_14":15,"Crop_15":16,"Crop_16":17,
      "Crop_17":18,"Crop_18":19,"Crop_19":20,"Crop_20":21,"Crop_21":22,"Crop_22":23,"Crop_23":24,"Crop_24":25,
      "Crop_25":26,"Crop_26":27,"Crop_27":28,"Crop_28":29,"Crop_29":30,"Crop_30":31,"Crop_31":32,"Crop_32":33,
      "Crop_33":34,"Crop_34":35,"Crop_35":36,"Crop_36":37,"Crop_37":38,"Crop_38":39,"Crop_39":40,"Crop_40":41,
      "Crop_41":42,"Crop_42":43,"Crop_43":44,"Crop_44":45,"Crop_45":46,"Crop_46":47,"Crop_47":48,"Crop_48":49,
      "Crop_49":50,"Crop_50":51,"Crop_51":52,"Crop_52":53,"Crop_53":54,"Crop_54":55,"Crop_55":56,"Crop_56":57,
      "Crop_57":58,"Crop_58":59,"Crop_59":60,"Crop_60":61,"Crop_61":62,"Crop_62":63,"Crop_63":64,"Crop_64":65,
      "Crop_65":66,"Crop_66":67,"Crop_67":68,"Crop_68":69,"Crop_69":70,"Crop_70":71,"Crop_71":72,"Crop_72":73,
      "Crop_73":74,"Crop_74":75,"Crop_75":76,"Crop_76":77,"Crop_77":78,"Crop_78":79,"Crop_79":80,"Crop_80":81,
      "Crop_81":82,"Crop_82":83,"Crop_83":84,"Crop_84":85,"Crop_85":86,"Crop_86":87,"Crop_87":88,"Crop_88":89,
      "Crop_89":90,"Crop_90":91,"Crop_91":92,"Crop_92":93,"Crop_93":94,"Crop_94":95,"Crop_95":96,"Crop_96":97,
      "Crop_97":98,"Crop_98":99,"Crop_99":100,"Crop_100":101,"Crop_101":102,"Crop_102":103,"Crop_103":104,"Crop_104":105,
      "Crop_105":106,"Crop_106":107,"Crop_107":108,"Crop_108":109,"Crop_109":110,"Crop_110":111,"Crop_111":112,"Crop_112":113,
      "Crop_113":114,"Crop_114":115,'Crop_115':116,'Crop_116':117}


choice=st.sidebar.selectbox('Select option',['Home','Graphs','Dataset','Model'])

# Home Page
if choice=='Home':
    st.title("""
    Agriculture Scenario of India.
    A Data Science Approach
    """)

    image = Image.open('Images/digital_agriculture3.jpg')

    st.image(image,caption='Agriculture and Data Science',use_column_width=True)



    with st.sidebar.beta_expander("Instruction"):
        st.write("""
            The chart above shows some numbers I picked for you.
            I rolled actual dice for these, so they're *guaranteed* to
            be random.
            """)
    st.header('Introduction')
    st.write("""
    India is an agriculture  based nation employing over 50% of the country’s workforce .
    However with the global and national food demand  on the rise due to the ever-increasing
    population, the agriculture sector is unable to meet the required productivity levels.
    Despite being called the backbone of Indian economy, the agriculture sector  has faced
    numerous setbacks in recent years .




    """)

    st.header('Purpose of The Project')
    st.write("""
    1) We want to do a comprehensive analysis which could act as **valuable reference material**.

    2) Very limited research on how weather pattern and climate change is affecting food market and **agriculture in India**.

    3) Proper analysis on how a more **generalized approach** can benefit agriculture.

    4) Remove the Deficiency of **good and comprehensive visualization** and interpretation of the entire problem statement.

    5) **Comparing Algorithms** best suited for the problem statement .



    """)
# Graph Page
elif choice=='Graphs':

    st.header('Graphs')

    image = Image.open('Images/predictive-analytics.jpg')

    st.image(image,caption='Agriculture and Data Science',use_column_width=True)


    st.subheader('State Wise Crop Production (1997-2018)')

    State_wise_crop=crop.groupby(['State_Name','Crop']).size().unstack().fillna(0)
    fig=px.bar(State_wise_crop)
    st.write(fig)

    st.subheader('State Wise Total Production (1997-2018)')

    crop_state_produce_units=crop.groupby('State_Name')['Production'].sum()
    fig=px.bar(crop_state_produce_units.sort_values())
    fig.update_layout(showlegend=False,width=800,height=600)
    st.write(fig)

    st.subheader('Total Crop Wise Production (1997-2018)')

    crop_p=crop
    crop_n=crop.loc[crop['Crop']== "Coconut "].index
    crop_p.drop(crop_n, inplace=True)
    crop_produce_index=crop_p.loc[crop_p['Production']==0].index
    crop_p.drop(crop_produce_index,inplace=True)

    crop_Tproduce=crop_p.groupby('Crop')['Production'].sum()
    fig=px.bar(crop_Tproduce.sort_values())
    fig.update_layout(showlegend=False,width=800,height=500)
    st.write(fig)

    st.subheader('Total Crop Production')

    crop_t=faostat[faostat['Year']>=1997 ]
    crop_t=crop_t.groupby('Year')['Value'].sum()
    fig=px.scatter(crop_t,labels={'value':"Production(tonnes)"})
    fig.update_layout(showlegend=False,width=800,height=500)
    fig.update_traces(mode='lines+markers')
    st.write(fig)

    st.subheader('Crop Production vs Avg. Rainfall')

    style.use('ggplot')

    rain_df=rain_df[rain_df["YEAR"]>=1997]

    fig, ax1 = plt.subplots(figsize=(15, 12))




    ax1.plot(crop_t,marker='D',mfc='green',ms='10',linewidth=3)


    color = 'tab:red'
    ax1.set_xlabel('Year',size=20)
    ax1.set_ylabel("Production (Tonnes )", color=color,size=15)

    ax1.tick_params(axis='y', labelcolor=color,labelsize=15)


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Rainfall in mm', color=color,size=15)  # we already handled the x-label with ax1
    ax2.plot(rain_df["YEAR"],rain_df["ANNUAL"], color=color,marker='o',mfc='purple',ms='10')
    ax2.tick_params(axis='y', labelcolor=color,labelsize=15)
    st.pyplot(fig=plt)

    st.subheader('Crop Production vs Avg. Temperature')

    #style.use('ggplot')

    temp_df=temp_df[temp_df["YEAR"]>=1997]


    fig, ax1 = plt.subplots(figsize=(15, 12))


    ax1.plot(crop_t,marker='D',mfc='green',ms='10',linewidth=3)


    color = 'tab:red'
    ax1.set_xlabel('Year',size=15)
    ax1.set_ylabel("Production (Tonnes )", color=color,size=15)

    ax1.tick_params(axis='y', labelcolor=color,labelsize=15)


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Temp (c)', color=color,size=15)  # we already handled the x-label with ax1
    ax2.plot(temp_df["YEAR"],temp_df["ANNUAL"], color=color,marker='o',mfc='purple',ms='10')
    ax2.tick_params(axis='y', labelcolor=color,labelsize=15)
    st.pyplot(fig=plt)


#Dataset Page
elif choice=='Dataset':

    st.title("""
    Dataset Used
    """)

    image = Image.open('Images/shutterstock_1333662722-678x381.jpg')

    st.image(image,caption='Agriculture and Data Science',use_column_width=True)
    # Dataset head
    st.header("""
    Dataset Used
    """)

    st.write(df.head(10))

    # Dataset Description
    st.write("""
    ## Dataset Description
    """)

    st.write(df.describe())

    # Dataset Columns
    st.write("""
    ## Dataset Columns
    """)

    st.write(df.columns)

#MODEL Training page
elif choice=='Model':
    def xgb_model():
        st.subheader('Code')
        st.code(
        """

            # Model Training

            df=pd.read_csv('Data/streamlit_data.csv')

            df_test=df.sample(frac = 0.1)

            X=df_test[['Crop_Year','Avg_Temp', 'Avg_Rain','states_id', 'crop_id', 'Area',]].values

            df_test.Production = df_test.Production.astype(int)

            y=df_test[['Production']].values

            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.70)

            clf = svm.SVC(kernel='linear') # Linear Kernel

            clf.fit(X_train,y_train.ravel())

            # Pickling the model

            pickle.dump(clf, open('svm_clf.pkl', 'wb'))




        """
        )
        # Reads in saved classification model
        import pickle
        xgb_load = pickle.load(open('rdg_model.pkl', 'rb'))


        # User Input Predictions

        st.subheader("User Input Predictions")
        u_pred=xgb_load.predict(df_user_input)
        st.write('Amount of Production Predicted : ' , u_pred[0]-230)

    def rgd_model():
        st.subheader('Code')
        st.code(
        """

            # Model Training

            X=df[['Crop_Year','Avg_Temp', 'Avg_Rain','states_id', 'crop_id', 'Area_10000',]].values

            df.Prod_1000000 = df.Prod_1000000.astype(int)

            y=df[['Prod_1000000']].values

            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.70)

            clf = RandomForestClassifier(n_estimators = 100)

            clf.fit(X_train,y_train.ravel())

            y_pred = clf.predict(X_test)

            st.subheader('Random Forest Classifier')
            st.write("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))




        """
        )
        # Reads in saved classification model
        import pickle
        load_clf = pickle.load(open('rdg_model.pkl', 'rb'))


        # User Input Predictions

        st.subheader("User Input Predictions")
        u_pred=load_clf.predict(df_user_input)
        st.write('Amount of Production Predicted : ' , u_pred[0])

    def gbr_model():
        st.subheader('Code')
        st.code(
        """

            # Model Training

            X=df[['Crop_Year','Avg_Temp', 'Avg_Rain','states_id', 'crop_id', 'Area_10000',]].values

            df.Prod_1000000 = df.Prod_1000000.astype(int)

            y=df[['Prod_1000000']].values

            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.70)

            clf = RandomForestClassifier(n_estimators = 100)

            clf.fit(X_train,y_train.ravel())

            y_pred = clf.predict(X_test)

            st.subheader('Random Forest Classifier')
            st.write("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))




        """
        )
        # Reads in saved classification model
        import pickle
        load_clf = pickle.load(open('gbr_model.pkl', 'rb'))


        # User Input Predictions

        st.subheader("User Input Predictions")
        u_pred=load_clf.predict(df_user_input)
        st.write('Amount of Production Predicted : ' , u_pred[0])



    def rfr_model():
        st.subheader('Code')
        st.code(
        """

            # Model Training

            X=df[['Crop_Year','Avg_Temp', 'Avg_Rain','states_id', 'crop_id', 'Area_10000',]].values

            df.Prod_1000000 = df.Prod_1000000.astype(int)

            y=df[['Prod_1000000']].values

            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.70)

            clf = RandomForestClassifier(n_estimators = 100)

            clf.fit(X_train,y_train.ravel())

            y_pred = clf.predict(X_test)

            st.subheader('Random Forest Classifier')
            st.write("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))




        """
        )
        # Reads in saved classification model
        import pickle
        import gzip, pickle, pickletools
        filepath = "random_forest_model.pkl"
        with gzip.open(filepath, 'rb') as f:
            p = pickle.Unpickler(f)
            load_clf= p.load()

        #load_clf = pickle.load(open('rfreg_model.pkl', 'rb'))


        # User Input Predictions

        st.subheader("User Input Predictions")
        u_pred=load_clf.predict(df_user_input)
        st.write('Amount of Production Predicted : ' , u_pred[0])


    st.title("""
    Model for Pridicting Production of Crops
    """)

    image = Image.open('Images/code_programming_text_141192_3840x2400.jpg')

    st.image(image,caption='Model Building',use_column_width=True)

    # Side Bar
    def user_input():

        st.sidebar.header('User Input Parameters')

        option_state=st.sidebar.selectbox('Select State',df.State_Name.unique())
        id_state=State_id[option_state]
        key_list=list(state_code.keys())
        val_list=list(state_code.values())
        position=val_list.index(id_state)
        code_value=key_list[position]
        st.sidebar.write('State selected is: ',option_state)


        option_Crop=st.sidebar.selectbox('Select Crop',df.Crop.unique())
        id_crop=crop_id[option_Crop]
        key_list_crop=list(crop_code.keys())
        val_list_crop=list(crop_code.values())
        position_crop=val_list_crop.index(id_crop)
        code_value_crop=key_list_crop[position_crop]
        st.sidebar.write('Crop selected is: ',option_Crop)


        option_year=st.sidebar.selectbox('Select Year',[1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012])
        st.sidebar.write('Year selected is: ',option_year)

        option_Area=st.sidebar.slider('Select Area',0.01,80000.0,1000.0)
        st.sidebar.write('Area selected is: ',option_Area)

        option_Temp=st.sidebar.slider('Select Temperature',24.10,25.13)
        st.sidebar.write('Temperature selected is: ',option_Temp)

        option_Rain=st.sidebar.slider('Select Rain Measure',920.80,1243.50)
        st.sidebar.write('Rain Measure selected is: ',option_Rain)


        # selected Parameters

        #data={'State_Name':id_state,
        #    'Crop':id_crop,
        #    'Year':option_year,
        #    'Area':option_Area,
        #    'Temp':option_Temp,
        #    'Rain':option_Rain}

        data={"State_Name_0":0,
      "State_Name_1":0,"State_Name_2":0,"State_Name_3":0,"State_Name_4":0,"State_Name_5":0,"State_Name_6":0,"State_Name_7":0,"State_Name_8":0,
      "State_Name_9":0,"State_Name_10":0,"State_Name_11":0,"State_Name_12":0,"State_Name_13":0,"State_Name_14":0,"State_Name_15":0,
      "State_Name_16":0,"State_Name_17":0,"State_Name_18":0,"State_Name_19":0,"State_Name_20":0,"State_Name_21":0,"State_Name_22":0,"State_Name_23":0,
      "State_Name_24":0,"State_Name_25":0,"State_Name_26":0,"State_Name_27":0,"State_Name_28":0,"State_Name_29":0,"State_Name_30":0,
      "Crop_0":0,"Crop_1":0,"Crop_2":0,"Crop_3":0,"Crop_4":0,"Crop_5":0,"Crop_6":0,"Crop_7":0,"Crop_8":0,
      "Crop_9":0,"Crop_10":0,"Crop_11":0,"Crop_12":0,"Crop_13":0,"Crop_14":0,"Crop_15":0,"Crop_16":0,
      "Crop_17":0,"Crop_18":0,"Crop_19":0,"Crop_20":0,"Crop_21":0,"Crop_22":0,"Crop_23":0,"Crop_24":0,
      "Crop_25":0,"Crop_26":0,"Crop_27":0,"Crop_28":0,"Crop_29":0,"Crop_30":0,"Crop_31":0,"Crop_32":0,
      "Crop_33":0,"Crop_34":0,"Crop_35":0,"Crop_36":0,"Crop_37":0,"Crop_38":0,"Crop_39":0,"Crop_40":0,
      "Crop_41":0,"Crop_42":0,"Crop_43":0,"Crop_44":0,"Crop_45":0,"Crop_46":0,"Crop_47":0,"Crop_48":0,
      "Crop_49":0,"Crop_50":0,"Crop_51":0,"Crop_52":0,"Crop_53":0,"Crop_54":0,"Crop_55":0,"Crop_56":0,
      "Crop_57":0,"Crop_58":0,"Crop_59":0,"Crop_60":0,"Crop_61":0,"Crop_62":0,"Crop_63":0,"Crop_64":0,
      "Crop_65":0,"Crop_66":0,"Crop_67":0,"Crop_68":0,"Crop_69":0,"Crop_70":0,"Crop_71":0,"Crop_72":0,
      "Crop_73":0,"Crop_74":0,"Crop_75":0,"Crop_76":0,"Crop_77":0,"Crop_78":0,"Crop_79":0,"Crop_80":0,
      "Crop_81":0,"Crop_82":0,"Crop_83":0,"Crop_84":0,"Crop_85":0,"Crop_86":0,"Crop_87":0,"Crop_88":0,
      "Crop_89":0,"Crop_90":0,"Crop_91":0,"Crop_92":0,"Crop_93":0,"Crop_94":0,"Crop_95":0,"Crop_96":0,
      "Crop_97":0,"Crop_98":0,"Crop_99":0,"Crop_100":0,"Crop_101":0,"Crop_102":0,"Crop_103":0,"Crop_104":0,
      "Crop_105":0,"Crop_106":0,"Crop_107":0,"Crop_108":0,"Crop_109":0,"Crop_110":0,"Crop_111":0,"Crop_112":0,
      "Crop_113":0,"Crop_114":0,'Crop_Year':option_year,'Avg_Temp':option_Temp, 'Avg_Rain':option_Rain, 'Area':option_Area}

        data[code_value]=1;
        data[code_value_crop]=1;
        features=pd.DataFrame(data,index=[0])
        return (features)

    df_user_input=user_input()

    st.subheader('User Input Parameters')

    st.write(df_user_input[['Crop_Year','Avg_Rain','Avg_Temp','Area']])

    st.subheader('Select Model')

    option_model=st.selectbox('Select Model',['RandomForest','XGBoost','Gradient Boosting','Ridge'])

    if st.button('RUN!!!!'):
        st.write('Running Model......')
        if option_model=='RandomForest' :
            rfr_model()
        elif option_model=='XGBoost':
            xgb_model()
        elif option_model=='Ridge':
            rgd_model()
        else:
            gbr_model()

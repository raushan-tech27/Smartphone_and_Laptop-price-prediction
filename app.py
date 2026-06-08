import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.title(" 💻📱 GadgetPrice Prediction System")
st.sidebar.title('Select One')
one = st.sidebar.selectbox('choose',['Laptop','Mobile'])


st.set_page_config(
    page_title="SmartPrice AI",
    layout="wide",
    page_icon="📱"
)

page_bg = """
<style>
.stApp {
background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSYhOxwDxGmnoj-136u5_gkk_GGOTI57GbXqA&s");
background-size: cover;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)


# import the model(smartphone)
df = pickle.load(open('df .pkl', 'rb'))
pipe = pickle.load(open('pipe .pkl', 'rb'))

# import the model(Laptop)
model= pickle.load(open('pipe1_laptop1.pkl', 'rb'))
df1= pickle.load(open('df_laptop1.pkl', 'rb'))


if one == 'Mobile':

    st.subheader('📱 SmartPhone Price Prediction')

    col1, col2 = st.columns(2)
    with col1:
        os = st.selectbox('OS', df['os'].unique())
        brand = st.selectbox('Brand', df['Brand'].unique())
        processor_brand = st.selectbox('Processor_Brand', df['processor_brand'].unique())
        core = st.selectbox('Core', df['core_type'].unique())
        camera = st.selectbox('Camera', df['rear_camera'].unique())
        x_res = st.number_input('x_resolution')

    with col2:
        Ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 18])
        Rom = st.selectbox('ROM(in GB)', [4, 8, 16, 32, 64, 128, 256, 512])
        premium = st.selectbox('Premium(Pro/Ultra/Plus/Max)', ['Yes', 'No'])
        is_5G = st.selectbox('5G', ['Yes', 'No'])
        extra_feature = st.selectbox('Extra_Feature(NFC/IR Blaster/Vo5G)', ['Yes', 'No'])
        y_res = st.number_input('y_resolution')

    processor_speed = st.slider('Processor_speed', 1.0, 5.0)
    screen_size = st.slider('Screen_Size', 1.0, 10.0)


    if st.button('Predict Price'):
        # query
        ppi = None
        if premium == 'Yes':
            premium = 1
        else:
            premium = 0

        if is_5G == 'Yes':
            is_5G = 1
        else:
            is_5G = 0

        if extra_feature == 'Yes':
            extra_feature = 1
        else:
            extra_feature = 0

        ppi = ((x_res ** 2) + (y_res ** 2)) ** 0.5 / screen_size

        query = np.array([os, brand, premium, is_5G, extra_feature, processor_brand, core, processor_speed,
                          camera, screen_size, Ram, Rom, ppi, x_res, y_res])

        query = query.reshape(1, 15)
        try:
            prediction = np.exp(pipe.predict(query)[0])

            st.success(f"Predicted Price: ₹{int(prediction):,}")

        except Exception as e:
            st.error(e)








elif one == 'Laptop':

    st.subheader(' 💻 Laptop Price Prediction')
    col1, col2 = st.columns(2)
    with col1:
        # brand
        company = st.selectbox('Brand', df1['Company'].unique())

        # type of laptop
        type = st.selectbox('Type', df1['TypeName'].unique())

        # Ram
        ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

        # weight
        weight = st.number_input('Weight of the Laptop')

        # Touchscreen
        touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

        # IPS
        ips = st.selectbox('IPS', ['No', 'Yes'])

    with col2:

        # resolution
        resolution = st.selectbox('Screen Resolution',
                                  ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800',
                                   '2560x1600',
                                   '2560x1440', '2304x1440'])

        # cpu
        cpu = st.selectbox('CPU', df1['Cpu brand'].unique())

        hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

        ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

        gpu = st.selectbox('GPU', df1['Gpu brand'].unique())

        os = st.selectbox('OS', df1['os'].unique())
    # screen size
    screen_size = st.slider('Screensize in inches', 10.0, 18.0, 13.0)

    if st.button('Predict Price'):
        # query
        ppi = None
        if touchscreen == 'Yes':
            touchscreen = 1
        else:
            touchscreen = 0

        if ips == 'Yes':
            ips = 1
        else:
            ips = 0

        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
        query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

        query = query.reshape(1, 12)
        try:
            prediction = np.exp(model.predict(query)[0])

            st.success(f"Predicted Price: ₹{int(prediction):,}")

        except Exception as e:
            st.error(e)








































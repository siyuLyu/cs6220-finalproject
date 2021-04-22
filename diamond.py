import streamlit as st
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from PIL import Image

# Loading model 
clf = joblib.load('./model/diamond-price-prediction-model.joblib')
cluster = joblib.load('./model/diamond-price-prediction-cluster.joblib')

# Create title and sidebar
st.title("Diamond Price Prediction Service")

st.write("##")

shape = Image.open('shape.jpg')
twod = Image.open('2-D.jpg')
threed = Image.open('3-D.jpg')
res = Image.open('predict.png')

depthlist = []
tablelist = []
caratlist = []
xlist = []
ylist = []
zlist = []

for i in np.arange(43.0, 79.0, 0.1):
    depthlist.append(round(i,1)) 

for i in range(43,96):
    tablelist.append(i)

for i in np.arange(0.20, 3.05,0.01):
    caratlist.append(round(i,2))

for i in np.arange(3.73, 9.55,0.01):
    xlist.append(round(i,2))

for i in np.arange(3.68, 31.9,0.01):
    ylist.append(round(i,2))

for i in np.arange(1.53, 5.65,0.01):
    zlist.append(round(i,2))

option1 = st.sidebar.selectbox(
    'Cut',
     ('Fair', 'Good', 'Ideal', 'Premium'))

# 'Cut:', option1

option2 = st.sidebar.selectbox(
    'Clarity',
     ('I1',	'IF', 'SI1', 'SI2',	'VS1', 'VS2', 'VVS1', 'VVS2'))

# 'Clarity:', option2

option3 = st.sidebar.selectbox(
    'Color',
     ('D','E','F','G','H','I','J'))

# 'Color:', option3

option4 = st.sidebar.selectbox(
    'Carat',
     (caratlist))

# 'Carat:', option4

option5 = st.sidebar.selectbox(
    'Table (The width of the diamond\'s table)',
     (tablelist))

# 'Table:', option5

option6 = st.sidebar.selectbox(
    'x',
     (xlist))

# 'x:', option6

option7 = st.sidebar.selectbox(
    'y',
     (ylist))

# 'y:', option7

option8 = st.sidebar.selectbox(
    'z ',
     (zlist))

# 'z:', option8

option9 = st.sidebar.selectbox(
    'Depth',
     (depthlist))

# 'Depth:', option9


parameter_list=['carat', 'depth', 'table', 'x', 'y', 'z', 'cut_Fair',
       'cut_Good', 'cut_Ideal', 'cut_Premium', 'cut_Very Good', 'color_D',
       'color_E', 'color_F', 'color_G', 'color_H', 'color_I', 'color_J',
       'clarity_I1', 'clarity_IF', 'clarity_SI1', 'clarity_SI2', 'clarity_VS1',
       'clarity_VS2', 'clarity_VVS1', 'clarity_VVS2']
parameter_input_values=[]
for i in range(26):
    parameter_input_values.append(0)

input_variables=pd.DataFrame([parameter_input_values],columns=parameter_list,dtype=float)
input_variables['carat'] = option4
input_variables['depth'] = option9
input_variables['table'] = option5
input_variables['x'] = option6
input_variables['y'] = option7
input_variables['z'] = option8

if option1 == 'Fair':
    input_variables['cut_Fair'] = 1
elif option1 == 'Good':
    input_variables['cut_Good'] = 1
elif option1 == 'Ideal':
    input_variables['cut_Ideal'] = 1
elif option1 == 'Premium':
    input_variables['cut_Premium'] = 1

if option2 == 'I1':
    input_variables['clarity_I1'] = 1
elif option2 == 'IF':
    input_variables['clarity_IF'] = 1
elif option2 == 'SI1':
    input_variables['clarity_SI1'] = 1
elif option2 == 'SI2':
    input_variables['clarity_SI2'] = 1
elif option2 == 'VS1':
    input_variables['clarity_VS1'] = 1
elif option2 == 'VVS1':
    input_variables['clarity_VVS1'] = 1
elif option2 == 'VVS2':
    input_variables['clarity_VVS2'] = 1

if option3 == 'D':
    input_variables['color_D'] = 1
elif option3 == 'E':
    input_variables['color_E'] = 1
elif option3 == 'F':
    input_variables['color_F'] = 1
elif option3 == 'G':
    input_variables['color_G'] = 1
elif option3 == 'H':
    input_variables['color_H'] = 1
elif option3 == 'I':
    input_variables['color_I'] = 1
elif option3 == 'J':
    input_variables['color_J'] = 1


# st.write(input_variables)




cluster_list=['carat',
       'clarity','price']
cluster_input_values=[]
for i in range(3):
    cluster_input_values.append(0)

input_cluster = pd.DataFrame([cluster_input_values],columns=cluster_list,dtype=float)

if option2 == 'I1':
    input_cluster['clarity'] = 0
elif option2 == 'IF':
    input_cluster['clarity'] = 1
elif option2 == 'SI1':
    input_cluster['clarity'] = 2
elif option2 == 'SI2':
    input_cluster['clarity'] = 3
elif option2 == 'VS1':
    input_cluster['clarity'] = 4
elif option2 == 'VS2':
    input_cluster['clarity'] = 5
elif option2 == 'VVS1':
    input_cluster['clarity'] = 6
elif option2 == 'VVS2':
    input_cluster['clarity'] = 7

input_cluster['carat'] = option4
# st.write(input_cluster)



if st.sidebar.button("Reveal Price"):
    prediction = clf.predict(input_variables)
    input_cluster['price'] = prediction[0]
    label = cluster.predict(input_cluster)
    
    # st.write(label[0])

    progress_bar = st.progress(0)
    status_text = st.empty()
    last_rows = np.random.randn(1, 1)
    # chart = st.line_chart(last_rows)
    chart = st.line_chart(input_variables)

    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        status_text.text("%i%% Complete" % i)
        chart.add_rows(new_rows)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(0.05)

    progress_bar.empty()

    st.write('Price $', prediction[0])
    st.write('#')
    st.image(res)
   
    if label[0] == 1 :
        st.write('Label: Fine Jewellry')
        st.write('#')
        # st.image(threed)
    else:
        st.write('Label: Economy')

else:
    st.image(shape) 


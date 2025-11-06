import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.write("Welcome")

st.header("Header")

name = st.text_input("Enter your name")
st.write(f'Hello, {name}!')

age = st.number_input('Enter your age', min_value=0, max_value=120, value=18)
st.write(f'You are {age} years old')

slider = st.slider('Select your slider level', min_value=1, max_value=10, value=3)
st.write(f'Slider: {slider}')

# Example data
df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [1, 3, 2, 4, 5],
    'z': [5, 4, 2, 3, 1]
})

fig = px.scatter_3d(df, x='x', y='y', z='z', title='Interactive 3D Scatter Plot')

fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        #bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='white'),
        yaxis=dict(showgrid=True, gridcolor='white'),
        zaxis=dict(showgrid=True, gridcolor='white'),
    ),
    paper_bgcolor='black',
    plot_bgcolor='black'
)

st.plotly_chart(fig)

if st.button('Say Hello!'):
    st.write('Hello guys!')

option = st.selectbox('Choose an option:', [1,2,3])
st.write(f'You select: {option}')

st.header('Displaying Data and Charts')

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 25]
}

df = pd.DataFrame(data)

st.subheader('Sample DataFrame')
st.write(df)

start_num = st.text_input('Enter a start number', value=0)
try:
    start_num = float(start_num)
except Exception as e:
    print('')

st.write(f'Hello, {start_num}!')
end_num = st.text_input('Enter a end number', value=10)
try:
    end_num = float(end_num)
except Exception as e:
    print('')
 
st.write(f'Hello, {end_num}!')


st.subheader('Matplotlib Chart')
fig, ax = plt.subplots()
x = np.linspace(start_num, end_num, 100)
ax.plot(x, np.sin(x))
st.pyplot(fig)

st.subheader('Columns Layout')
col1, col2 = st.columns(2)

with col1:
    st.write('Content for col1')
with col2:
    st.write('Content for col2')

st.subheader('Tabs layout')
tab1, tab2 = st.tabs(['Tab 1', 'Tab 2'])

with tab1:
    st.write('Content for tab1')
with tab2:
    st.write('Content for tab2')

#st.sidebar.title('Sidebar Title')
#sidebar_option = st.sidebar.selectbox('Select an option', ['A', 'B', 'C'])
#st.sidebar.write(f'You selected: {sidebar_option}')

col_main, col_side = st.columns([8, 4], gap="small")  # ≈ 66% / 33%

with col_main:
    a = st.text_input('Enter a', value=10)

with col_side:
    b = st.text_input('Enter b', value=10)

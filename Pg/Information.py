import streamlit as st
import pandas as pd

# Assume your dataset is loaded here
# Replace this with your actual dataset loading code
# Example dataset
data = {
    'Worker_ID': ['W1', 'W2', 'W3'],
    'Production_Volume': [100, 150, 200],
    'Date': ['2024-06-01', '2024-06-02', '2024-06-03']
}
df = pd.DataFrame(data)

# Streamlit app
st.title('Worker ID and Production Volume Entry')

# Input for Worker ID
worker_id = st.text_input("Enter the Worker ID:")

# Input for Production Volume
production_volume = st.number_input("Enter the Production Volume:", min_value=0)

# Button to display dataset
if st.button('Show Dataset'):
    st.subheader('Filtered Dataset')
    
    # Filter dataset based on Worker ID and Production Volume
    filtered_df = df[(df['Worker_ID'] == worker_id) & (df['Production_Volume'] == production_volume)]
    
    # Display filtered dataset
    st.write(filtered_df)

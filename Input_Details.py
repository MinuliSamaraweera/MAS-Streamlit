import streamlit as st
import csv
from datetime import date, timedelta
import re

def calculate_total_defects(run_off_d1, open_seam_d2, spi_errors_d3, high_low_d4):
    return run_off_d1 + open_seam_d2 + spi_errors_d3 + high_low_d4

def calculate_count(production_value, defect_count):
    return production_value - defect_count 

def record_exists(worker_id, date_str):
    csv_file = '/Users/minu/Desktop/MAS-Streamlit/Dataset/updated_worker_defect_details.csv'
    
    try:
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Worker_ID'] == worker_id and row['Date'] == date_str:
                    return True
    except FileNotFoundError:
        return False
    
    return False

def update_defect_counts(worker_id, date_str, shift, run_off_d1, open_seam_d2, spi_errors_d3, high_low_d4, production_value):
    csv_file = '/Users/minu/Desktop/MAS-Streamlit/Dataset/updated_worker_defect_details.csv'
    
    defect_count = calculate_total_defects(run_off_d1, open_seam_d2, spi_errors_d3, high_low_d4)
    count = calculate_count(production_value, defect_count)
    
    new_data = {
        'Worker_ID': worker_id,
        'Date': date_str,
        'Run_Off_D1': run_off_d1,
        'Open_Seam_D2': open_seam_d2,
        'SPI_Errors_D3': spi_errors_d3,
        'High_Low_D4': high_low_d4,
        'Production_Volume': production_value,
        'Shift': shift,
        'defect_count': defect_count,
        'count': count
    }
    
    try:
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Worker_ID', 'Date', 'Run_Off_D1', 'Open_Seam_D2', 
                                                      'SPI_Errors_D3', 'High_Low_D4', 'Production_Volume', 
                                                      'Shift', 'defect_count', 'count'])
            
            # Check if file is empty and write headers
            file.seek(0, 2)  # Go to end of file
            if file.tell() == 0:
                writer.writeheader()
            
            writer.writerow(new_data)
        
        st.success("Data successfully submitted.")
    except Exception as e:
        st.error(f"Error occurred: {e}")

def main():
    st.title('Defect Counts Input')
    
    # Input fields
    worker_id = st.text_input('Enter Worker ID (e.g., W_XXXXX):')
    
    # Worker ID format validation
    if not re.match(r'^W_\d{5}$', worker_id):
        st.warning('Worker ID should be in the format W_ followed by exactly 5 digits (e.g., W_XXXXX)')
    
    # Date selection restricted to today and previous day
    max_date = date.today()
    min_date = max_date - timedelta(days=1)  # Today and previous one day
    date_selected = st.date_input('Select Date:', min_value=min_date, max_value=max_date, value=max_date)
    
    shift = st.selectbox('Select Shift:', options=['Morning', 'Afternoon', 'Evening'])
    run_off_d1 = st.number_input('Run_Off_D1 count:', min_value=0)
    open_seam_d2 = st.number_input('Open_Seam_D2 count:', min_value=0)
    spi_errors_d3 = st.number_input('SPI_Errors_D3 count:', min_value=0)
    high_low_d4 = st.number_input('High_Low_D4 count:', min_value=0)
    production_value = st.number_input('Production Volume:', min_value=0)
    
    # Button to submit data
    if st.button('Submit'):
        if re.match(r'^W_\d{5}$', worker_id):
            date_str = date_selected.strftime('%m/%d/%y')  # Changed to match format e.g., 1/1/24
            if not record_exists(worker_id, date_str):
                update_defect_counts(worker_id, date_str, shift, 
                                     int(run_off_d1), int(open_seam_d2), int(spi_errors_d3), int(high_low_d4), production_value)
            else:
                st.warning(f'Record already exists for Worker ID {worker_id} on date {date_str}.')
        else:
            st.warning('Please enter a valid Worker ID format (e.g., W_XXXXX)')

if __name__ == '__main__':
    main()
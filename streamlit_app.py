import streamlit as st
import pandas as pd

from src.RULBattery.pipelines.pip_06_prediction_pipeline import CustomData, PredictionPipeline

# Function to display the homepage
def show_homepage():
    st.title("Welcome to Battery RUL Prediction App")
    st.write("""
        This web application will help you predict the Remaining Useful Life (RUL) of a battery based on its current condition.
    """)

# Function to display the prediction page
def show_prediction_page():
    st.title("Battery RUL Prediction")

    # Create form for user input
    with st.form(key='prediction_form'):
        cycle_index = st.text_input('Cycle Index', value='0')
        discharge_time_s = st.text_input('Discharge Time (s)', value='0')
        decrement_3_6_3_4v_s = st.text_input('Decrement 3.6-3.4V (s)', value='0')
        max_voltage_discharge_v = st.text_input('Max. Voltage Discharge (V)', value='0')
        min_voltage_charge_v = st.text_input('Min. Voltage Charge (V)', value='0')
        time_at_4_15v_s = st.text_input('Time at 4.15V (s)', value='0')
        time_constant_current_s = st.text_input('Time Constant Current (s)', value='0')
        charging_time_s = st.text_input('Charging Time (s)', value='0')

        # Submit button
        submit_button = st.form_submit_button(label='Predict')

    # Handle form submission
    if submit_button:
        try:
            # Prepare form data
            form_data = {
                'cycle_index': cycle_index,
                'discharge_time_s': discharge_time_s,
                'decrement_3_6_3_4v_s': decrement_3_6_3_4v_s,
                'max_voltage_discharge_v': max_voltage_discharge_v,
                'min_voltage_charge_v': min_voltage_charge_v,
                'time_at_4_15v_s': time_at_4_15v_s,
                'time_constant_current_s': time_constant_current_s,
                'charging_time_s': charging_time_s
            }

            # Create an instance of the CustomData class
            custom_data = CustomData(**form_data)

            # Convert the form data dictionary to a dataframe
            pred_df = custom_data.get_data_as_dataframe()

            # Initialize the prediction pipeline
            prediction_pipeline = PredictionPipeline()

            # Get the prediction
            prediction = prediction_pipeline.make_predictions(pred_df)

            # Display the result
            st.subheader("Prediction Result")
            st.write(f"The predicted remaining useful life of the battery is: {prediction[0]}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Main function to run the app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Predict"])

    if page == "Home":
        show_homepage()
    elif page == "Predict":
        show_prediction_page()

if __name__ == '__main__':
    main()

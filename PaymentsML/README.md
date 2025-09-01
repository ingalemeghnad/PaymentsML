# Payment Anomaly Detection

This project implements a Payment Anomaly Detection system using machine learning techniques. The application is built with Streamlit and allows users to upload payment transaction data to identify potential anomalies.

## Project Structure

- `app.py`: Main application code for the Payment Anomaly Detection demo.
- `global_iforest.pkl`: Serialized model file for the global isolation forest model.
- `debtor_iforest.pkl`: Serialized model file for the debtor-level isolation forest model.
- `debtor_profiles.csv`: Contains aggregated profiles for debtors to enhance scoring.
- `requirements.txt`: Lists the Python dependencies required for the project.
- `README.md`: Documentation for the project.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd PaymentsML
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

- Upload a CSV file containing payment transactions through the Streamlit interface.
- The application will process the data, perform anomaly detection, and display the results, including flagged transactions and visualizations.

## License

This project is licensed under the MIT License.
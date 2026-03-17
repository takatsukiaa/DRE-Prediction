import sqlite3
import pandas as pd


def fetch_raw_clinical_data(db_name="PatientClinicalData.db"):
    """
    Extract raw clinical data from SQLite and transform it into a wide-format DataFrame.
    This function performs data retrieval and merging without trend calculations.
    """
    # Establish connection to the SQLite database
    conn = sqlite3.connect(db_name)

    # 1. Fetch static patient metadata
    # Contains: patient_id, gender, dob, age_of_onset, dre, etc.
    query_patients = "SELECT * FROM patients"
    df_patients = pd.read_sql_query(query_patients, conn)

    # 2. Fetch dynamic clinical features
    # Original schema: (patient_id, feature_name, time_point, value)
    query_features = "SELECT * FROM clinical_features"
    df_features = pd.read_sql_query(query_features, conn)

    # 3. Transform data from Long Format to Wide Format (Pivoting)
    # Each row will represent a unique (patient_id, time_point) pair.
    # Columns will correspond to the individual feature names.
    df_wide = df_features.pivot_table(
        index=['patient_id', 'time_point'],
        columns='feature_name',
        values='value'
    ).reset_index()

    # 4. Merge static metadata with dynamic features
    # Joins the patient-level info (gender, dre status) to each time-point slice.
    df_final = pd.merge(df_wide, df_patients, on='patient_id', how='left')
    selected_columns = ['patient_id', 'time_point', 'ddd_load', 'Seizure', 'dre']
    df_final = df_final[selected_columns]
    # Close the database connection
    conn.close()

    return df_final


if __name__ == "__main__":
    try:
        # Execute data extraction
        raw_dataset = fetch_raw_clinical_data()

        # Display the structure of the retrieved data
        print("Data extraction successful.")
        print(f"Total slices (rows): {len(raw_dataset)}")
        print(f"Features (columns): {raw_dataset.columns.tolist()}")

        # Preview the first few rows
        print("\nDataset Preview:")
        print(raw_dataset.head())

    except Exception as e:
        print(f"An error occurred during extraction: {e}")
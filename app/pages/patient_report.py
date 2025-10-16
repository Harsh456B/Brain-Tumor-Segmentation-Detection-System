import streamlit as st

def patient_report_page():
    st.title("Patient Report")
    st.write("Your complete diagnostic report will be available here after the MRI detection.")

    # Simulate a report generation (You can replace this with an actual report generation logic)
    report_content = """
    **Patient Name**: John Doe
    **MRI Scan Date**: 2025-04-22
    **Diagnosis**: No tumor detected
    **Remarks**: MRI scan appears normal. No signs of tumor detected in the scan.
    """

    st.text_area("Patient Report", report_content, height=300)

    # Generate a downloadable report
    st.download_button("Download Report", report_content, file_name="patient_report.txt")

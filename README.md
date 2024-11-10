

# ECG Data Analysis Dashboard

### Project Summary
This project focuses on the **qualitative analysis of ECG (electrocardiogram) signals** using advanced deep learning methods. The analysis aims to support the detection and study of heart rate variability (HRV) and associated cardiac anomalies. Using Streamlit for interactive visualizations, this dashboard allows users to upload ECG files and conduct signal processing, HRV metrics calculations, and anomaly detection. This study is particularly important for advancing healthcare technology in Senegal and assessing cardiovascular health risks.

---

## 1. Project Context and Importance
In Senegal and many parts of the world, access to advanced medical technology for detecting and diagnosing cardiovascular diseases is limited. Cardiovascular diseases often go undetected due to insufficient resources, leading to adverse outcomes. This study introduces a **machine learning-based approach to analyze ECG signals** to detect heart-related anomalies more efficiently. By deploying AI tools in ECG analysis, we can enhance early diagnosis, which can aid in effective intervention and improve patient outcomes.

## 2. Objectives
This project aims to:
- **Develop and validate** a machine learning and signal processing tool for ECG analysis.
- **Identify HRV metrics** and assess cardiac anomalies from ECG signals.
- **Visualize and interpret** HRV metrics to assist healthcare professionals.
- **Provide a comprehensive dashboard** to interactively explore ECG data, focusing on HRV, stress indices, and potential anomalies like bradycardia or tachycardia.

## 3. Project Impact
The impact of this project is:
- **Early Diagnosis**: Assisting in early detection of heart-related anomalies, potentially reducing cardiovascular complications.
- **Accessibility**: Providing a tool for clinicians in resource-limited settings.
- **Scalability**: With more data, this tool could be refined for broader application across West African healthcare.

## 4. Key Features and Results
This project is powered by **Streamlit**, allowing interactive exploration of ECG data, including:
- **Uploading ECG Data**: Load `.mat` format ECG data files for analysis.
- **Filtering ECG Signals**: Bandpass filtering and noise reduction for signal clarity.
- **Detecting R-peaks and HRV Metrics**: Identify heartbeats, calculate HRV metrics (e.g., RMSSD, SDNN), and display the results.
- **Anomaly Detection**: Using HRV metrics to identify possible anomalies like bradycardia, tachycardia, and signs of stress.
- **Detailed Visualization**: Display filtered ECG signals, detected R peaks, and HRV metrics over time.

## 5. Results and Observations
Based on the HRV metrics and anomaly detection algorithms, we observed:
- **Classification of Heart Rate Variability (HRV) anomalies**: Enabling clinicians to gauge overall heart health and stress levels.
- **Detection of common ECG anomalies**: Bradycardia, Tachycardia, and potential indicators of cardiovascular strain.
- **Patient-specific customization**: HRV thresholds adjust based on individual factors like age, activity level, weight, and sex.

---

## 6. How to Run the Project Locally

### Prerequisites
Ensure you have the following installed on your machine:
- **Python 3.8+**
- **Streamlit** (for the web application)
- **NumPy**, **Pandas**, **SciPy** (for data processing)
- **Altair** (for visualization)
- **Scikit-Learn**, **TensorFlow** (for deep learning components, if required)

### Installation Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ecg-dashboard.git
   cd ecg-dashboard
   ```

2. **Install dependencies**:
   You can use `pip` to install all required packages. Run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Application**:
   Start the Streamlit server by running:
   ```bash
   streamlit run main.py
   ```

4. **Upload ECG Files**:
   - Once the dashboard opens, you can upload `.mat` files containing ECG data.
   - Select the preprocessing parameters (margin, target length), then click **Run Preprocessing** to start the analysis.
   - View ECG signals, HRV metrics, and detected anomalies interactively.

### Project Structure
- **main.py**: Contains the Streamlit dashboard code.
- **final.py**: Implements the ECG processing functions, HRV calculations, and anomaly detection logic.
- **requirements.txt**: Lists the required Python packages.

### Note
Ensure `.mat` files are structured properly with keys like `ECG_1`, `ECG_2`, `beatpos`, etc., as expected by the `load_single_ecg` function in `final.py`.

---

## 7. Future Work
In future iterations, we aim to:
- **Enhance Anomaly Detection**: Refine and expand the anomaly categories.
- **Integrate More ECG Leads**: Support for additional ECG lead data for comprehensive analysis.
- **Optimize Model for Faster Computation**: Improve the deep learning model's efficiency to handle larger datasets.

---

This project provides a foundation for automated ECG analysis, aiming to reduce the diagnostic gap in Senegal and beyond. The application can be expanded to include real-time monitoring, more sophisticated anomaly detection, and data from more diverse populations, enhancing its utility in resource-limited settings.

---

Feel free to contribute, suggest enhancements, or reach out with questions!

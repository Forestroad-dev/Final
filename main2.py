import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from final import load_single_ecg, bandpass_filter, preprocess_and_filter_ecg2, calculate_vfc_metrics, calculate_overall_quality, etiquetage_ecg_vfc
from scipy.signal import find_peaks

# Streamlit dashboard layout
st.title("ECG Data Analysis Dashboard")

# File uploader for ECG data
uploaded_file = st.file_uploader("Upload ECG data file (MAT format)", type="mat")
if uploaded_file is not None:
    # Convert the uploaded file to a BytesIO object
    file_data = BytesIO(uploaded_file.read())

    # Process the uploaded file with `load_single_ecg`
    ecg_data, df_characteristics, beatpos = load_single_ecg(file_data)

    if ecg_data is not None:
        # Display patient characteristics
        st.subheader("Patient Characteristics")
        st.write(df_characteristics)

        # Display ECG data shape and basic info
        st.subheader("ECG Data Shape")
        st.write(f": {ecg_data.shape}")

        # # Show the first few rows of `beatpos` if available
        # st.subheader("Beat Positions (R Peaks)")
        # st.write(beatpos[:10] if beatpos is not None else "No beat positions detected")
        
       # Select a single lead (e.g., Lead 2) from ecg_data
        # Assuming ecg_data is already loaded
        selected_lead = 1  # Lead 2 is the focus

        # Step 1: Apply bandpass filter to the selected lead signal
        filtered_ecg_data = bandpass_filter(ecg_data[selected_lead], lowcut=0.5, highcut=30, fs=250)

        # Step 2: Detect R-peaks
        height_threshold = np.mean(filtered_ecg_data) + 1.5 * np.std(filtered_ecg_data)
        distance_samples = int(0.6 * 250)
        r_peaks, _ = find_peaks(filtered_ecg_data, height=height_threshold, distance=distance_samples)

        # Display only the first 5,000 samples to manage memory usage
        max_samples = 15000
        filtered_segment = filtered_ecg_data[:max_samples]
        r_peaks_segment = r_peaks[r_peaks < max_samples]

        # Prepare data for visualization
        time_axis = np.arange(len(filtered_segment)) / 250 / 60  # Convert samples to time in minutes
        r_peak_values = np.full_like(filtered_segment, np.nan)
        r_peak_values[r_peaks_segment] = filtered_segment[r_peaks_segment]

        # Combine time, ECG signal, and R-peak data into a DataFrame
        ecg_df = pd.DataFrame({
            "Time (min)": time_axis,
            "ECG Signal": filtered_segment,
            "R Peaks": r_peak_values
        })

        # Step 3: Create an Altair plot
        ecg_line = alt.Chart(ecg_df).mark_line(color="blue").encode(
            x=alt.X("Time (min):Q", title="Time (minutes)"),
            y=alt.Y("ECG Signal:Q", title="Amplitude")
        )

        r_peaks_points = alt.Chart(ecg_df).mark_point(color="red", size=60).encode(
            x="Time (min):Q",
            y="R Peaks:Q"
        )

        # Combine both layers
        ecg_chart = (ecg_line + r_peaks_points).properties(
            title="Filtered ECG Signal with Detected R Peaks (First 5,000 Samples)",
            width=800,
            height=400
        )

        # Display the chart
        st.altair_chart(ecg_chart, use_container_width=True)
               
        # Add margin and target length inputs
        st.subheader("Preprocessing Parameters")
        margin_minutes = st.number_input("Margin (minutes)", min_value=1, max_value=10, value=2)
        target_length_minutes = st.number_input("Target Length (minutes)", min_value=1, max_value=10, value=5)

        # Run preprocessing on button click
        if st.button("Run Preprocessing"):
            margin_samples = int(margin_minutes * 60 * 250)
            target_samples = int(target_length_minutes * 60 * 250)
            processed_ecg, processed_beatpos = preprocess_and_filter_ecg2(
                ecg_data, beatpos, target_length=target_samples
            )

            # Display the shape of processed ECG data
            st.subheader("Processed ECG Shape")
            st.write(f": {processed_ecg.shape}")

            # Visualize each lead in processed ECG
            st.subheader("Processed ECG Preview (First 5000 Samples of Each Lead)")
            for lead in range(processed_ecg.shape[0]):
                st.write(f"Lead {lead + 1}")
                st.line_chart(processed_ecg[lead][:5000], height=300, width=800)
            
            rr_intervals = np.diff(beatpos) * 1000
            if processed_ecg is not None and rr_intervals is not None:
                # Call the function to calculate quality scores
                lead_scores, lead_sqis, lead_combined_scores, overall_quality, overall_sqi, overall_combined_score = calculate_overall_quality(processed_ecg, rr_intervals)
                            
                    # Find the best lead quality (highest combined score)
                best_lead_quality = max(lead_combined_scores)
                best_lead_index = lead_combined_scores.index(best_lead_quality) + 1  # +1 for 1-based lead indexing

                            # Display the calculated scores in Streamlit
                st.subheader("Signal Quality Metrics")
                st.write("**Quality Score for Each Lead:**", lead_scores)
                st.write("**SQI for Each Lead:**", lead_sqis)
                st.write("**Combined Quality Score for Each Lead:**", lead_combined_scores)
                st.write("**Overall Signal Quality:**", overall_quality)
                st.write("**Overall SQI of the Signal:**", overall_sqi)
                st.write("**Overall Combined Signal Quality Score:**", overall_combined_score)
                            
                            # Display best lead quality
                st.write(f"**Best Lead Quality (Lead {best_lead_index}):** {best_lead_quality}")         
                             
            # Calculate RR intervals in ms from beat positions
            sampling_rate = 250
            rr_intervals = np.diff(beatpos) * 1000 # Assuming sampling rate = 250 Hz
            # Calculate and display VFC metrics
            vfc_metrics = calculate_vfc_metrics(rr_intervals)
            st.subheader("VFC Metrics")
            st.write(pd.DataFrame([vfc_metrics]))  # Display metrics as a DataFrame

            time_axis = (processed_beatpos[1:] / 1000).flatten()  # Flatten to ensure 1D and convert to seconds

            # Ensure lengths match
            if len(rr_intervals) == len(time_axis):
                # Create the DataFrame
                tachogram_df = pd.DataFrame({"Time (s)": time_axis, "RR Interval (ms)": rr_intervals})

                # Plot Tachogram with Altair
                st.subheader("Tachogram (RR Intervals over Time)")
                tachogram_chart = alt.Chart(tachogram_df).mark_line(point=alt.OverlayMarkDef(size=40, color="red")).encode(
                    x=alt.X("Time (s):Q", title="Time (seconds)"),
                    y=alt.Y("RR Interval (ms):Q", title="RR Interval (ms)")
                ).properties(
                    width=800,
                    height=400,
                    title="Tachogram of RR Intervals"
                )

                # Display the chart
                st.altair_chart(tachogram_chart, use_container_width=True)
            else:
                st.error("Mismatch in lengths between RR intervals and time axis. Unable to display Tachogram.")

        # Step 1: Filter the ECG signal from the selected lead
                ecg_filtered = bandpass_filter(ecg_data[selected_lead], lowcut=0.5, highcut=40, fs=250)
                
                # Step 2: Detect R peaks with adjusted threshold and minimum distance
                height_threshold = np.mean(ecg_filtered) + 0.5 * np.std(ecg_filtered)
                distance_samples = int(0.6 * 250)
                peaks, _ = find_peaks(ecg_filtered, height=height_threshold, distance=distance_samples)
                
                # Step 3: Calculate RR intervals in ms and filter out short intervals
                rr_intervals = np.diff(peaks) * (1000 / 250)  # Convert intervals to milliseconds
                rr_intervals_filtered = rr_intervals[rr_intervals > 300]  # Filter out intervals < 300 ms
                
                # Step 4: Calculate HRV metrics
                hrv_metrics = calculate_vfc_metrics(rr_intervals_filtered)
                
                # Display HRV metrics as a DataFrame
                st.subheader("HRV Metrics")
                st.write(pd.DataFrame([hrv_metrics]))  # Display HRV metrics

                # Step 5: Plot the filtered ECG signal with R peaks
                st.subheader("Filtered ECG Signal with Detected R Peaks")
                time_axis = np.arange(len(ecg_filtered)) / 250 / 60  # Convert to minutes for x-axis
                ecg_df = pd.DataFrame({"Time (min)": time_axis, "ECG Signal": ecg_filtered})
                ecg_df["R Peaks"] = np.nan
                ecg_df.loc[peaks, "R Peaks"] = ecg_filtered[peaks]

                st.line_chart(ecg_df.set_index("Time (min)")[["ECG Signal", "R Peaks"]].dropna(), width=800)
                
                # Patient information input
                st.subheader("Patient Information")

                # Sexe selection
                sexe = st.selectbox("Sexe", ["H", "F"], index=0)

                # Numeric inputs for poids, taille, and age
                poids = st.number_input("Poids (kg)", min_value=30, max_value=200, value=70)
                taille = st.number_input("Taille (cm)", min_value=100, max_value=250, value=170)
                age = st.number_input("Âge", min_value=0, max_value=120, value=30)

                # Dropdown for physical activity level
                activite_physique = st.selectbox("Niveau d'activité physique", ["élevée", "modérée", "sédentaire"], index=1)

                                # Button to calculate anomaly detection based on HRV metrics
                if st.button("Run Anomaly Detection"):
                            # Run anomaly detection using the patient information and HRV metrics
                    etiquetage_results_df = etiquetage_ecg_vfc(vfc_metrics, sexe, poids, taille, activite_physique, age)

                            # Display detected anomalies
                    st.subheader("Anomalies Detected")
                    st.write(etiquetage_results_df)

    else:
        st.error("Failed to load ECG data. Please check the file format and content.")

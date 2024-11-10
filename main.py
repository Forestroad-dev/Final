import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from io import BytesIO
from scipy.signal import find_peaks
from final import (
    load_single_ecg, 
    bandpass_filter, 
    preprocess_and_filter_ecg2, 
    calculate_vfc_metrics, 
    calculate_overall_quality, 
    etiquetage_ecg_vfc
)

# Streamlit dashboard layout
st.title("ECG Data Analysis Dashboard")

# File uploader for ECG data
uploaded_file = st.file_uploader("Upload ECG data file (MAT format)", type="mat")
if uploaded_file is not None:
    file_data = BytesIO(uploaded_file.read())  # Convert uploaded file to BytesIO object

    # Load ECG data and characteristics
    ecg_data, df_characteristics, beatpos = load_single_ecg(file_data)

    if ecg_data is not None:
        st.subheader("Patient Characteristics")
        st.write(df_characteristics)

        st.subheader("ECG Data Shape")
        st.write(f"Shape: {ecg_data.shape}")

        # Select Lead and filter for visualization
        selected_lead = 1
        filtered_ecg_data = bandpass_filter(ecg_data[selected_lead], lowcut=0.5, highcut=30, fs=250)

        # Detect R-peaks
        height_threshold = np.mean(filtered_ecg_data) + 1.5 * np.std(filtered_ecg_data)
        distance_samples = int(0.6 * 250)
        r_peaks, _ = find_peaks(filtered_ecg_data, height=height_threshold, distance=distance_samples)

        # Visualize filtered ECG and R-peaks
        def plot_ecg_with_peaks(data, peaks, title="Filtered ECG with R-peaks"):
            time_axis = np.arange(len(data)) / 250 / 60  # Time in minutes
            ecg_df = pd.DataFrame({"Time (min)": time_axis, "ECG Signal": data})
            ecg_df["R Peaks"] = np.nan
            ecg_df.loc[peaks, "R Peaks"] = data[peaks]
            st.line_chart(ecg_df.set_index("Time (min)")[["ECG Signal", "R Peaks"]].dropna(), width=800)

        # st.subheader("Filtered ECG Signal with Detected R Peaks (First 5,000 Samples)")
        # plot_ecg_with_peaks(filtered_ecg_data[:5000], r_peaks[r_peaks < 5000])

        # Preprocessing section with margin and target length inputs
        st.subheader("Preprocessing Parameters")
        margin_minutes = st.number_input("Margin (minutes)", min_value=1, max_value=10, value=2)
        target_length_minutes = st.number_input("Target Length (minutes)", min_value=1, max_value=10, value=5)
        
        # Convert margin and target length from minutes to samples
        fs = 250  # Sampling frequency in Hz
        margin_samples = int(margin_minutes * 60 * fs)
        target_samples = int(target_length_minutes * 60 * fs)

        # Select the segment based on margin and target length
        start_sample = margin_samples
        end_sample = start_sample + target_samples
        filtered_segment = filtered_ecg_data[start_sample:end_sample]

        # Detect R-peaks within the selected segment
        r_peaks_segment = r_peaks[(r_peaks >= start_sample) & (r_peaks < end_sample)] - start_sample

        # Prepare data for visualization
        time_axis = np.arange(len(filtered_segment)) / fs / 60  # Convert samples to time in minutes
        r_peak_values = np.full_like(filtered_segment, np.nan)
        r_peak_values[r_peaks_segment] = filtered_segment[r_peaks_segment]

        # Combine time, ECG signal, and R-peak data into a DataFrame
        ecg_df = pd.DataFrame({
            "Time (min)": time_axis,
            "ECG Signal": filtered_segment,
            "R Peaks": r_peak_values
        })

        # Create an Altair plot
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
            title=f"Filtered ECG Signal with Detected R Peaks ({target_length_minutes} Minutes Segment)",
            width=800,
            height=400
        )

        # Display the chart
        st.altair_chart(ecg_chart, use_container_width=True)
               
        # Add margin and target length inputs
        st.subheader("Preprocessing Parameters")
        
        if st.button("Run Preprocessing"):
            margin_samples = int(margin_minutes * 60 * 250)
            target_samples = int(target_length_minutes * 60 * 250)
            processed_ecg, processed_beatpos = preprocess_and_filter_ecg2(ecg_data, beatpos, target_length=target_samples)

            st.subheader("Processed ECG Shape")
            st.write(f"Shape: {processed_ecg.shape}")

            # Display the first 5,000 samples of each lead
            st.subheader("Processed ECG Preview (First 5000 Samples of Each Lead)")
            for lead in range(processed_ecg.shape[0]):
                st.write(f"Lead {lead + 1}")
                st.line_chart(processed_ecg[lead][:5000], height=300, width=800)

            # Quality Metrics Calculation
            
                        # Flatten beatpos if it's multi-dimensional
            if beatpos is not None and beatpos.ndim > 1:
                beatpos = beatpos.flatten()

            # Display the beat positions and their count
            st.subheader("Beat Positions (beatpos)")

            # Show the first few beat positions and total length
            st.write("First few beat positions:", beatpos[:10])
            st.write("Total number of beat positions:", len(beatpos))

            sampling_rate=250
            rr_intervals_test = np.diff(beatpos) * 1000  # Assuming sampling rate = 250 Hz
            lead_scores, lead_sqis, lead_combined_scores, overall_quality, overall_sqi, overall_combined_score = calculate_overall_quality(processed_ecg, rr_intervals_test)

            best_lead_quality = max(lead_combined_scores)
            best_lead_index = lead_combined_scores.index(best_lead_quality) + 1

            # Display quality metrics
            st.subheader("Signal Quality Metrics")
            st.write("Quality Score for Each Lead:", lead_scores)
            st.write("SQI for Each Lead:", lead_sqis)
            st.write("Combined Quality Score for Each Lead:", lead_combined_scores)
            st.write("Overall Signal Quality:", overall_quality)
            st.write("Overall SQI of the Signal:", overall_sqi)
            st.write("Overall Combined Signal Quality Score:", overall_combined_score)
            st.write(f"Best Lead Quality (Lead {best_lead_index}): {best_lead_quality}")

            # VFC Metrics Calculation
            vfc_metrics = calculate_vfc_metrics(rr_intervals_test)
            st.session_state['vfc_metrics'] = vfc_metrics
            st.subheader("VFC Metrics")
            st.write(pd.DataFrame([vfc_metrics]))

            # Tachogram Visualization
            # Prepare time axis for tachogram visualization
            time_axis = (beatpos[1:] / 1000).flatten()  # Convert beatpos to seconds, adjust for RR intervals length
            
            # Check if RR intervals and time axis match in length before plotting
            if len(rr_intervals_test) == len(time_axis):
                # Create a DataFrame for the tachogram
                tachogram_df = pd.DataFrame({
                    "Time (s)": time_axis,
                    "RR Interval (ms)": rr_intervals_test
                })

                # Plot Tachogram with Altair
                tachogram_chart = alt.Chart(tachogram_df).mark_line(
                    point=alt.OverlayMarkDef(size=40, color="red")
                ).encode(
                    x=alt.X("Time (s):Q", title="Time (seconds)"),
                    y=alt.Y("RR Interval (ms):Q", title="RR Interval (ms)")
                ).properties(
                    width=800, height=400, title="Tachogram of RR Intervals"
                )
                
                # Display the tachogram chart
                st.subheader("Tachogram (RR Intervals over Time)")
                st.altair_chart(tachogram_chart, use_container_width=True)
            else:
                st.error("Mismatch in lengths between RR intervals and time axis. Unable to display Tachogram.")
                
            ecg_filtered = bandpass_filter(ecg_data[selected_lead], lowcut=0.5, highcut=40, fs=250)
                
                # Step 2: Detect R peaks with adjusted threshold and minimum distance
            height_threshold = np.mean(ecg_filtered) + 0.5 * np.std(ecg_filtered)
            distance_samples = int(0.5 * 250)
            peaks, _ = find_peaks(ecg_filtered, height=height_threshold, distance=distance_samples)
            rr_intervals = np.diff(peaks) * (1000 / 250)  # Convert intervals to milliseconds
            rr_intervals_filtered = rr_intervals[rr_intervals > 300]  # Filter out intervals < 300 ms
            
            hrv_metrics = calculate_vfc_metrics(rr_intervals_filtered)
            st.session_state['hrv_metrics'] = hrv_metrics  # Store in session state
                
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

        # Patient Information and Anomaly Detection
        st.subheader("Patient Information")
        sexe = st.selectbox("Sexe", ["H", "F"], index=0)
        poids = st.number_input("Poids (kg)", min_value=30, max_value=200, value=70)
        taille = st.number_input("Taille (cm)", min_value=100, max_value=250, value=170)
        age = st.number_input("Âge", min_value=0, max_value=120, value=30)
        activite_physique = st.selectbox("Niveau d'activité physique", ["élevée", "modérée", "sédentaire"], index=1)

        # Ensure VFC metrics are available
        
        if 'vfc_metrics' in st.session_state:
        # Perform anomaly detection for VFC metrics
            etiquetage_vfc_df = etiquetage_ecg_vfc(
                pd.DataFrame([st.session_state['vfc_metrics']]), sexe, poids, taille, activite_physique, age
            )
            st.write("VFC Metrics - État et Anomalies")
            st.write(etiquetage_vfc_df)
        else:
            st.error("VFC metrics not available. Please run preprocessing and VFC metric calculation first.")
                    
        if 'hrv_metrics' in st.session_state:
            etiquetage_results_df = etiquetage_ecg_vfc(pd.DataFrame([st.session_state['hrv_metrics']]), sexe, poids, taille, activite_physique, age)
            st.subheader("Anomalies Detected")
            st.write(etiquetage_results_df)
        else:
            st.error("Please run preprocessing and VFC metric calculation first.")
            


    else:
        st.error("Failed to load ECG data. Please check the file format and content.")

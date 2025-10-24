import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go

import cv_parser
import viz
import db


st.set_page_config(page_title="Floorplan CV Viewer", layout="wide")

st.title("Floorplan CV Viewer")
st.markdown("Upload and visualize object detection results from floor plan CV models.")

db.init_db()

if 'df' not in st.session_state:
    st.session_state.df = None
if 'raw_json' not in st.session_state:
    st.session_state.raw_json = None
if 'source_name' not in st.session_state:
    st.session_state.source_name = None
if 'current_run_id' not in st.session_state:
    st.session_state.current_run_id = None

with st.sidebar:
    st.header("Controls")
    
    with st.expander("Load Sample JSON"):
        if st.button("Load Sample"):
            try:
                with open('sample/1_friendly_cv.json', 'r') as f:
                    sample_json = f.read()
                st.session_state.raw_json = sample_json
                data = cv_parser.load_json(sample_json)
                st.session_state.source_name = "sample"
                st.session_state.current_run_id = None
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample: {str(e)}")
    
    uploaded_file = st.file_uploader("Upload JSON", type=['json'])
    
    if uploaded_file is not None:
        st.session_state.raw_json = uploaded_file.read().decode('utf-8')
        st.session_state.source_name = uploaded_file.name
        st.session_state.current_run_id = None
    
    saved_runs = db.list_runs()
    if saved_runs:
        st.divider()
        run_options = {f"{r.source_name} ({r.created_at.strftime('%Y-%m-%d %H:%M')})": r.id for r in saved_runs}
        selected_run = st.selectbox("Load Saved Run", options=["Current"] + list(run_options.keys()))
        
        if selected_run != "Current" and selected_run in run_options:
            run_id = run_options[selected_run]
            if st.session_state.current_run_id != run_id:
                st.session_state.current_run_id = run_id
                loaded_df = db.load_detections(run_id)
                st.session_state.df = loaded_df
                for run in saved_runs:
                    if run.id == run_id:
                        st.session_state.raw_json = run.raw_json
                        st.session_state.source_name = run.source_name
                        break
    
    st.divider()
    
    assume_center = not st.checkbox("x,y are top-left (not center)", value=False)
    
    if st.session_state.raw_json:
        try:
            data = cv_parser.load_json(st.session_state.raw_json)
            df = cv_parser.parse_detections(data, assume_center=assume_center)
            st.session_state.df = df
        except Exception as e:
            st.error(f"Error parsing JSON: {str(e)}")
            st.session_state.df = None
    
    if st.session_state.df is not None and not st.session_state.df.empty:
        st.divider()
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        all_classes = sorted(st.session_state.df['class_name'].unique().tolist())
        selected_classes = st.multiselect(
            "Filter by Class",
            options=all_classes,
            default=all_classes
        )
        
        show_labels = st.checkbox("Show Labels", value=True)
        
        st.divider()
        
        if st.button("Save to DB", type="primary"):
            try:
                run = db.create_run(
                    source_name=st.session_state.source_name or "unknown",
                    raw_json=st.session_state.raw_json
                )
                num_saved = db.bulk_insert_detections(run.id, st.session_state.df)
                st.success(f"Saved {num_saved} detections to database!")
                st.session_state.current_run_id = run.id
            except Exception as e:
                st.error(f"Error saving to database: {str(e)}")
        
        if st.button("Clear Current Session"):
            st.session_state.df = None
            st.session_state.raw_json = None
            st.session_state.source_name = None
            st.session_state.current_run_id = None
            st.rerun()

if st.session_state.df is None or st.session_state.df.empty:
    st.info("Upload a JSON file or load the sample to get started.")
else:
    df = st.session_state.df
    
    df_filtered = df[
        (df['confidence'] >= confidence_threshold) &
        (df['class_name'].isin(selected_classes))
    ]
    
    tab1, tab2, tab3 = st.tabs(["Visualizer", "Detections Table", "Summary"])
    
    with tab1:
        st.subheader("Floor Plan Detections")
        
        if df_filtered.empty:
            st.warning("No detections match the current filters.")
        else:
            canvas_w, canvas_h = cv_parser.infer_canvas_size(df_filtered)
            
            fig = viz.make_figure(canvas_w, canvas_h)
            viz.add_boxes(fig, df_filtered, show_labels=show_labels)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption(f"Showing {len(df_filtered)} of {len(df)} detections")
    
    with tab2:
        st.subheader("Detection Data")
        
        if df_filtered.empty:
            st.warning("No detections match the current filters.")
        else:
            display_df = df_filtered[[
                'class_name', 'confidence', 'x', 'y', 'width', 'height',
                'x1', 'y1', 'x2', 'y2', 'area', 'detection_uuid'
            ]].copy()
            
            display_df = display_df.sort_values('confidence', ascending=False)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="detections.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.subheader("Summary Statistics")
        
        if df_filtered.empty:
            st.warning("No detections match the current filters.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Detections", len(df_filtered))
            with col2:
                st.metric("Mean Confidence", f"{df_filtered['confidence'].mean():.3f}")
            with col3:
                st.metric("Min Confidence", f"{df_filtered['confidence'].min():.3f}")
            with col4:
                st.metric("Max Confidence", f"{df_filtered['confidence'].max():.3f}")
            
            st.divider()
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Count by Class")
                class_counts = df_filtered['class_name'].value_counts().reset_index()
                class_counts.columns = ['Class', 'Count']
                
                color_map = viz.class_color_map(df_filtered)
                colors = [color_map.get(cls, '#000000') for cls in class_counts['Class']]
                
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=class_counts['Class'],
                        y=class_counts['Count'],
                        marker_color=colors
                    )
                ])
                fig_bar.update_layout(
                    xaxis_title="Class",
                    yaxis_title="Count",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col_b:
                st.subheader("Confidence Distribution")
                fig_hist = go.Figure(data=[
                    go.Histogram(
                        x=df_filtered['confidence'],
                        nbinsx=20,
                        marker_color='#3498db'
                    )
                ])
                fig_hist.update_layout(
                    xaxis_title="Confidence",
                    yaxis_title="Frequency",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            st.divider()
            
            st.subheader("Classes Present")
            classes_present = df_filtered['class_name'].unique()
            st.write(", ".join(sorted(classes_present)))

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go

import cv_parser
import viz
import db
import ocr_parser


st.set_page_config(page_title="Floorplan CV Viewer", layout="wide")

st.title("Floorplan CV Viewer")
st.markdown("Upload and visualize object detection results from floor plan CV models.")

db.init_db()

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'raw_json' not in st.session_state:
    st.session_state.raw_json = None
if 'source_name' not in st.session_state:
    st.session_state.source_name = None
if 'current_run_id' not in st.session_state:
    st.session_state.current_run_id = None
if 'ocr_df' not in st.session_state:
    st.session_state.ocr_df = None
if 'ocr_raw_json' not in st.session_state:
    st.session_state.ocr_raw_json = None
if 'ocr_source_name' not in st.session_state:
    st.session_state.ocr_source_name = None
if 'current_ocr_run_id' not in st.session_state:
    st.session_state.current_ocr_run_id = None
# Wall visibility controls
if 'wall_visibility' not in st.session_state:
    st.session_state.wall_visibility = {}
# OCR control settings
if 'show_ocr_boxes' not in st.session_state:
    st.session_state.show_ocr_boxes = False
if 'ocr_confidence_threshold' not in st.session_state:
    st.session_state.ocr_confidence_threshold = 50.0
if 'ocr_text_size' not in st.session_state:
    st.session_state.ocr_text_size = 8
if 'show_ocr_words' not in st.session_state:
    st.session_state.show_ocr_words = False

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
        # Use simpler labels to avoid tooltip issues
        run_labels = ["Current"] + [r.source_name for r in saved_runs]
        run_map = {r.source_name: r for r in saved_runs}
        
        selected_label = st.selectbox("Load Saved Run", options=run_labels)
        
        if selected_label != "Current" and selected_label in run_map:
            run = run_map[selected_label]
            if st.session_state.current_run_id != run.id:
                st.session_state.current_run_id = run.id
                loaded_df = db.load_detections(run.id)
                st.session_state.df = loaded_df
                st.session_state.raw_json = run.raw_json
                st.session_state.source_name = run.source_name
                # Show timestamp info separately
                st.caption(f"Saved: {run.created_at.strftime('%Y-%m-%d %H:%M')}")
    
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
        
        st.divider()
        st.subheader("Class Visibility")
        
        all_classes = sorted(st.session_state.df['class_name'].unique().tolist())
        
        # Initialize class visibility states if not present
        for class_name in all_classes:
            if f"class_{class_name}" not in st.session_state:
                st.session_state[f"class_{class_name}"] = True
        
        # Callback functions for toggle buttons
        def show_all_classes():
            for class_name in all_classes:
                st.session_state[f"class_{class_name}"] = True
        
        def hide_all_classes():
            for class_name in all_classes:
                st.session_state[f"class_{class_name}"] = False
        
        # Quick toggle buttons with callbacks
        col1, col2 = st.columns(2)
        with col1:
            st.button("Show All", key="show_all_classes", on_click=show_all_classes)
        with col2:
            st.button("Hide All", key="hide_all_classes", on_click=hide_all_classes)
        
        # Create columns for checkboxes (2 columns for better layout)
        cols = st.columns(2)
        selected_classes = []
        
        for i, class_name in enumerate(all_classes):
            col_idx = i % 2
            with cols[col_idx]:
                # Use the session state value for the checkbox
                if st.checkbox(class_name, key=f"class_{class_name}"):
                    selected_classes.append(class_name)
        
        # Update session state with currently selected classes
        st.session_state.selected_classes = selected_classes
        
        st.divider()
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
            st.session_state.wall_visibility = {}
            st.rerun()

if st.session_state.df is None or st.session_state.df.empty:
    st.info("Upload a JSON file or load the sample to get started.")
else:
    df = st.session_state.df
    
    # Use selected classes from session state for filtering
    if 'selected_classes' not in st.session_state:
        st.session_state.selected_classes = df['class_name'].unique().tolist()
    
    df_filtered = df[
        (df['confidence'] >= confidence_threshold) &
        (df['class_name'].isin(st.session_state.selected_classes))
    ]
    
    tab1, tab2, tab3, tab4 = st.tabs(["Visualizer", "Detections Table", "Summary", "OCR"])
    
    with tab1:
        st.subheader("Floor Plan Detections")
        
        if df_filtered.empty:
            st.warning("No detections match the current filters.")
        else:
            canvas_w, canvas_h = cv_parser.infer_canvas_size(df_filtered)

            wall_mask = df_filtered['class_name'].str.lower() == 'wall'
            wall_detections = df_filtered.loc[wall_mask, [
                'detection_uuid',
                'detection_name',
                'wall_direction',
                'confidence'
            ]].copy()

            current_wall_ids = set(wall_detections['detection_uuid'].tolist())
            stored_visibility = st.session_state.wall_visibility

            stale_ids = [wall_id for wall_id in list(stored_visibility.keys())
                         if wall_id not in current_wall_ids]
            for wall_id in stale_ids:
                stored_visibility.pop(wall_id, None)

            for wall_id in current_wall_ids:
                widget_key = f"wall_visibility_{wall_id}"
                if widget_key in st.session_state:
                    stored_visibility[wall_id] = bool(st.session_state[widget_key])
                else:
                    stored_visibility.setdefault(wall_id, True)

            hidden_wall_ids = [
                wall_id for wall_id, is_visible in stored_visibility.items()
                if not is_visible
            ]

            if hidden_wall_ids:
                df_visible = df_filtered[~df_filtered['detection_uuid'].isin(hidden_wall_ids)]
            else:
                df_visible = df_filtered

            fig = viz.make_figure(canvas_w, canvas_h)
            viz.add_boxes(fig, df_visible, show_labels=show_labels)

            # Add OCR overlay if available
            if st.session_state.ocr_df is not None and not st.session_state.ocr_df.empty:
                ocr_scaled = ocr_parser.scale_ocr_to_detections(
                    st.session_state.ocr_df, canvas_w, canvas_h
                )
                # Filter OCR data based on settings
                ocr_filtered = ocr_scaled[ocr_scaled['confidence'] >= st.session_state.ocr_confidence_threshold]
                if not st.session_state.show_ocr_words:
                    ocr_filtered = ocr_filtered[ocr_filtered['block_type'] == 'LINE']
                
                viz.add_ocr_text(fig, ocr_filtered, 
                               show_boxes=st.session_state.show_ocr_boxes, 
                               text_size=st.session_state.ocr_text_size,
                               min_confidence=st.session_state.ocr_confidence_threshold/100)

            st.plotly_chart(fig, use_container_width=True)

            st.caption(f"Showing {len(df_visible)} of {len(df)} detections")

            if not wall_detections.empty:
                st.divider()
                st.markdown("#### Wall Visibility")

                wall_detections = wall_detections.sort_values('detection_name')
                num_columns = min(3, len(wall_detections)) or 1
                wall_cols = st.columns(num_columns)

                for idx, wall in enumerate(wall_detections.itertuples(index=False)):
                    col = wall_cols[idx % num_columns]
                    wall_label = wall.detection_name
                    if getattr(wall, 'wall_direction', ''):
                        wall_label = f"{wall_label} ({wall.wall_direction})"

                    checkbox_key = f"wall_visibility_{wall.detection_uuid}"
                    default_value = stored_visibility.get(wall.detection_uuid, True)
                    is_visible = col.checkbox(wall_label, value=default_value, key=checkbox_key)
                    stored_visibility[wall.detection_uuid] = is_visible

    with tab2:
        st.subheader("Detection Data")

        if df_filtered.empty:
            st.warning("No detections match the current filters.")
        else:
            columns = [
                'detection_name', 'class_name', 'confidence', 'x', 'y', 'width', 'height',
                'wall_direction', 'x1', 'y1', 'x2', 'y2', 'detection_uuid'
            ]

            available_columns = [col for col in columns if col in df_filtered.columns]

            display_df = df_filtered[available_columns].copy()
            
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
    
    with tab4:
        st.subheader("OCR Text Extraction")
        st.markdown("Upload AWS Textract JSON output to overlay text on the floor plan visualization.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Sample loader
            with st.expander("Load Sample OCR JSON"):
                if st.button("Load OCR Sample"):
                    try:
                        with open('sample/sample_ocr.json', 'r') as f:
                            ocr_json = f.read()
                        st.session_state.ocr_raw_json = ocr_json
                        st.session_state.ocr_source_name = "sample_ocr"
                        st.session_state.current_ocr_run_id = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading OCR sample: {str(e)}")
            
            # File uploader
            ocr_file = st.file_uploader("Upload AWS Textract JSON", type=['json'], key="ocr_uploader")
            
            if ocr_file is not None:
                st.session_state.ocr_raw_json = ocr_file.read().decode('utf-8')
                st.session_state.ocr_source_name = ocr_file.name
                st.session_state.current_ocr_run_id = None
            
            # Saved OCR runs
            saved_ocr_runs = db.list_ocr_runs()
            if saved_ocr_runs:
                # Use simpler labels to avoid tooltip issues
                ocr_run_labels = ["Current"] + [r.source_name for r in saved_ocr_runs]
                ocr_run_map = {r.source_name: r for r in saved_ocr_runs}
                
                selected_ocr_label = st.selectbox(
                    "Load Saved OCR Run", 
                    options=ocr_run_labels,
                    key="ocr_run_select"
                )
                
                if selected_ocr_label != "Current" and selected_ocr_label in ocr_run_map:
                    ocr_run = ocr_run_map[selected_ocr_label]
                    if st.session_state.current_ocr_run_id != ocr_run.id:
                        st.session_state.current_ocr_run_id = ocr_run.id
                        loaded_ocr_df = db.load_ocr_blocks(ocr_run.id)
                        st.session_state.ocr_df = loaded_ocr_df
                        st.session_state.ocr_raw_json = ocr_run.raw_json
                        st.session_state.ocr_source_name = ocr_run.source_name
                        # Show timestamp info separately
                        st.caption(f"Saved: {ocr_run.created_at.strftime('%Y-%m-%d %H:%M')}")
        
        with col2:
            st.subheader("OCR Controls")
            
            st.session_state.show_ocr_boxes = st.checkbox(
                "Show OCR Bounding Boxes", 
                value=st.session_state.show_ocr_boxes,
                key="ocr_boxes_check"
            )
            st.session_state.ocr_confidence_threshold = st.slider(
                "OCR Confidence Threshold",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.ocr_confidence_threshold,
                step=5.0,
                key="ocr_confidence"
            )
            st.session_state.ocr_text_size = st.slider(
                "Text Size", 
                min_value=6, 
                max_value=20, 
                value=st.session_state.ocr_text_size,
                key="text_size_slider"
            )
            st.session_state.show_ocr_words = st.checkbox(
                "Show Words (not just lines)", 
                value=st.session_state.show_ocr_words,
                key="show_words_check"
            )
        
        # Parse OCR if available
        if st.session_state.ocr_raw_json:
            try:
                canvas_w, canvas_h = cv_parser.infer_canvas_size(df_filtered)
                ocr_data = ocr_parser.load_textract_json(st.session_state.ocr_raw_json)
                metadata = ocr_parser.get_document_metadata(ocr_data)
                ocr_df = ocr_parser.parse_text_blocks(ocr_data, canvas_w, canvas_h)
                st.session_state.ocr_df = ocr_df
                
                # Display metadata
                st.divider()
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Pages", metadata.get('pages', 1))
                with col_m2:
                    st.metric("Text Lines", metadata.get('text_lines', 0))
                with col_m3:
                    st.metric("Words", metadata.get('words', 0))
                
                # Filter OCR data
                ocr_filtered = ocr_df[ocr_df['confidence'] >= st.session_state.ocr_confidence_threshold]
                if not st.session_state.show_ocr_words:
                    ocr_filtered = ocr_filtered[ocr_filtered['block_type'] == 'LINE']
                
                # Display OCR visualization
                st.divider()
                st.subheader("OCR Text Overlay")
                
                fig_ocr = viz.make_figure(canvas_w, canvas_h)
                viz.add_boxes(fig_ocr, df_filtered, show_labels=False)
                viz.add_ocr_text(fig_ocr, ocr_filtered, 
                               show_boxes=st.session_state.show_ocr_boxes, 
                               text_size=st.session_state.ocr_text_size,
                               min_confidence=st.session_state.ocr_confidence_threshold/100)
                
                st.plotly_chart(fig_ocr, use_container_width=True)
                
                # OCR data table
                st.divider()
                st.subheader("OCR Text Data")
                
                display_ocr = ocr_filtered[['text', 'block_type', 'confidence', 
                                           'x1', 'y1', 'x2', 'y2']].copy()
                display_ocr = display_ocr.sort_values('confidence', ascending=False)
                st.dataframe(display_ocr, use_container_width=True, hide_index=True)
                
                # Save OCR to DB
                if st.button("Save OCR to DB", type="primary", key="save_ocr"):
                    try:
                        ocr_run = db.create_ocr_run(
                            source_name=st.session_state.ocr_source_name or "unknown_ocr",
                            raw_json=st.session_state.ocr_raw_json,
                            pages=metadata.get('pages', 1)
                        )
                        num_saved = db.bulk_insert_ocr_blocks(ocr_run.id, st.session_state.ocr_df)
                        st.success(f"Saved {num_saved} OCR text blocks to database!")
                        st.session_state.current_ocr_run_id = ocr_run.id
                    except Exception as e:
                        st.error(f"Error saving OCR to database: {str(e)}")
                
                # Clear OCR session
                if st.button("Clear OCR Session", key="clear_ocr"):
                    st.session_state.ocr_df = None
                    st.session_state.ocr_raw_json = None
                    st.session_state.ocr_source_name = None
                    st.session_state.current_ocr_run_id = None
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error parsing OCR JSON: {str(e)}")
                st.session_state.ocr_df = None
        else:
            st.info("Upload an AWS Textract JSON file or load the sample to get started with OCR.")
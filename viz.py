import plotly.graph_objects as go
import pandas as pd
from typing import Dict


def class_color_map(df: pd.DataFrame) -> Dict[str, str]:
    """
    Create deterministic color mapping for classes.
    
    Args:
        df: DataFrame with class_name column
        
    Returns:
        Dictionary mapping class names to colors
    """
    default_colors = {
        'door': '#3498db',
        'window': '#2ecc71',
        'wall': '#95a5a6',
    }
    
    unique_classes = df['class_name'].unique()
    color_map = {}
    
    fallback_colors = ['#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
    fallback_idx = 0
    
    for cls in unique_classes:
        if cls in default_colors:
            color_map[cls] = default_colors[cls]
        else:
            color_map[cls] = fallback_colors[fallback_idx % len(fallback_colors)]
            fallback_idx += 1
    
    return color_map


def make_figure(canvas_w: float, canvas_h: float) -> go.Figure:
    """
    Create a Plotly figure with proper canvas settings.
    
    Args:
        canvas_w: Canvas width
        canvas_h: Canvas height
        
    Returns:
        Configured Plotly Figure
    """
    fig = go.Figure()
    
    fig.update_xaxes(
        range=[0, canvas_w],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        title=None
    )
    
    fig.update_yaxes(
        range=[canvas_h, 0],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        # Removed scaleanchor and scaleratio to allow free-form zoom
        # scaleanchor="x",  # This was forcing fixed proportions
        # scaleratio=1,     # This was maintaining 1:1 aspect ratio
        title=None
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        margin=dict(l=10, r=10, t=10, b=10),
        height=600,
        hovermode='closest'
    )
    
    return fig


def add_boxes(fig: go.Figure, df_filtered: pd.DataFrame, show_labels: bool = False):
    """
    Add bounding boxes to the figure.
    
    Args:
        fig: Plotly Figure to add boxes to
        df_filtered: Filtered DataFrame with detections
        show_labels: Whether to show labels on boxes
    """
    if df_filtered.empty:
        return
    
    color_map = class_color_map(df_filtered)
    
    for _, row in df_filtered.iterrows():
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        class_name = row['class_name']
        detection_name = row.get('detection_name', class_name)
        confidence = row['confidence']
        color = color_map.get(class_name, '#000000')
        
        fig.add_shape(
            type="rect",
            x0=x1, y0=y1,
            x1=x2, y1=y2,
            line=dict(color=color, width=2),
            fillcolor=color,
            opacity=0.3
        )
        
        if show_labels:
            label_text = f"{detection_name} ({confidence:.2f})"
            fig.add_annotation(
                x=x1,
                y=y1,
                text=label_text,
                showarrow=False,
                xanchor='left',
                yanchor='bottom',
                bgcolor=color,
                font=dict(color='white', size=10),
                opacity=0.8
            )


def add_ocr_text(fig: go.Figure, ocr_df: pd.DataFrame, show_boxes: bool = False,
                 min_confidence: float = 0.5, text_size: int = 8):
    """
    Add OCR text overlay to the figure.
    
    Args:
        fig: Plotly Figure to add text to
        ocr_df: DataFrame with OCR text blocks
        show_boxes: Whether to show bounding boxes around text
        min_confidence: Minimum confidence threshold for displaying text
        text_size: Font size for text display
    """
    if ocr_df.empty:
        return
    
    # Filter by confidence
    df_filtered = ocr_df[ocr_df['confidence'] >= min_confidence]
    
    # Only show LINE blocks by default (not individual WORD blocks to avoid clutter)
    df_lines = df_filtered[df_filtered['block_type'] == 'LINE']
    
    for _, row in df_lines.iterrows():
        text = row['text']
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        center_x = row['center_x']
        center_y = row['center_y']
        confidence = row['confidence']
        
        # Add text annotation
        fig.add_annotation(
            x=center_x,
            y=center_y,
            text=text,
            showarrow=False,
            xanchor='center',
            yanchor='middle',
            font=dict(color='#e74c3c', size=text_size, family='monospace'),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#e74c3c',
            borderwidth=1,
            opacity=0.9
        )
        
        # Optionally show bounding box
        if show_boxes:
            fig.add_shape(
                type="rect",
                x0=x1, y0=y1,
                x1=x2, y1=y2,
                line=dict(color='#e74c3c', width=1, dash='dot'),
                fillcolor='rgba(231, 76, 60, 0.1)',
                opacity=0.5
            )

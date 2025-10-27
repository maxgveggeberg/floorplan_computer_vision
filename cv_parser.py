import json
import pandas as pd
from typing import Tuple

from shapely.geometry import LineString

from geometry_utils import add_detection_names, add_wall_direction


def load_json(raw_bytes_or_str) -> dict:
    """
    Load JSON from bytes or string.
    
    Args:
        raw_bytes_or_str: Either bytes from file upload or string
        
    Returns:
        Parsed JSON as dictionary
    """
    if isinstance(raw_bytes_or_str, bytes):
        return json.loads(raw_bytes_or_str.decode('utf-8'))
    elif isinstance(raw_bytes_or_str, str):
        return json.loads(raw_bytes_or_str)
    else:
        return raw_bytes_or_str


def parse_detections(data: dict, assume_center: bool = True) -> pd.DataFrame:
    """
    Parse detection data from JSON into a pandas DataFrame.
    
    Args:
        data: Dictionary with 'predictions' key containing list of detections
        assume_center: If True, x,y are box centers (YOLO-style).
                      If False, x,y are top-left corners.
    
    Returns:
        DataFrame with parsed detections including computed corners
    """
    if 'predictions' not in data:
        raise ValueError("JSON must contain 'predictions' key")
    
    predictions = data['predictions']
    if not predictions:
        raise ValueError("No predictions found in JSON")
    
    rows = []
    required_keys = ['x', 'y', 'width', 'height', 'confidence', 'class', 'class_id', 'detection_id']
    
    for pred in predictions:
        try:
            if not all(key in pred for key in required_keys):
                continue
            
            x = float(pred['x'])
            y = float(pred['y'])
            width = float(pred['width'])
            height = float(pred['height'])
            confidence = float(pred['confidence'])
            class_name = str(pred['class'])
            class_id = int(pred['class_id'])
            detection_uuid = str(pred['detection_id'])
            
            if assume_center:
                x1 = x - width / 2
                y1 = y - height / 2
                x2 = x + width / 2
                y2 = y + height / 2
            else:
                x1 = x
                y1 = y
                x2 = x + width
                y2 = y + height
            
            area = width * height
            
            rows.append({
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'confidence': confidence,
                'class_name': class_name,
                'class_id': class_id,
                'detection_uuid': detection_uuid,
                'area': area
            })
        except (KeyError, ValueError, TypeError):
            continue
    
    if not rows:
        raise ValueError("No valid detections could be parsed from JSON")
    
    df = pd.DataFrame(rows)
    df = add_detection_names(df)
    df = add_wall_direction(df)

    line_start_x = []
    line_start_y = []
    line_end_x = []
    line_end_y = []
    line_geometries = []

    for row in df.itertuples(index=False):
        if str(row.class_name).lower() != 'wall':
            line_start_x.append(None)
            line_start_y.append(None)
            line_end_x.append(None)
            line_end_y.append(None)
            line_geometries.append(None)
            continue

        direction = getattr(row, 'wall_direction', '')
        center_x = row.x
        center_y = row.y

        start_point: Tuple[float, float] | None = None
        end_point: Tuple[float, float] | None = None

        if direction == 'vertical':
            half_height = row.height / 2
            start_point = (center_x, center_y + half_height)
            end_point = (center_x, center_y - half_height)
        elif direction == 'horizontal':
            half_width = row.width / 2
            start_point = (center_x - half_width, center_y)
            end_point = (center_x + half_width, center_y)

        if start_point and end_point:
            line_start_x.append(start_point[0])
            line_start_y.append(start_point[1])
            line_end_x.append(end_point[0])
            line_end_y.append(end_point[1])
            line_geometries.append(LineString([start_point, end_point]))
        else:
            line_start_x.append(None)
            line_start_y.append(None)
            line_end_x.append(None)
            line_end_y.append(None)
            line_geometries.append(None)

    df['wall_line_start_x'] = line_start_x
    df['wall_line_start_y'] = line_start_y
    df['wall_line_end_x'] = line_end_x
    df['wall_line_end_y'] = line_end_y
    df['wall_linestring'] = line_geometries

    return df


def infer_canvas_size(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Infer canvas size from detection coordinates with padding.
    
    Args:
        df: DataFrame with x2, y2 columns
        
    Returns:
        Tuple of (width, height) for canvas
    """
    if df.empty:
        return 1000, 1000
    
    max_x = df['x2'].max()
    max_y = df['y2'].max()
    
    padding_factor = 1.1
    
    canvas_width = max_x * padding_factor
    canvas_height = max_y * padding_factor
    
    return canvas_width, canvas_height

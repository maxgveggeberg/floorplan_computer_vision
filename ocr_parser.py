import json
import pandas as pd
from typing import Dict, List, Tuple


def load_textract_json(raw_bytes_or_str) -> dict:
    """
    Load AWS Textract JSON from bytes or string.
    
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


def parse_text_blocks(data: dict, canvas_width: float = 1200, canvas_height: float = 900) -> pd.DataFrame:
    """
    Parse text blocks from AWS Textract JSON into a pandas DataFrame.
    
    Args:
        data: Dictionary from AWS Textract with 'Blocks' key
        canvas_width: Width of the original detection coordinate space
        canvas_height: Height of the original detection coordinate space
    
    Returns:
        DataFrame with parsed text blocks including scaled coordinates
    """
    if 'Blocks' not in data:
        raise ValueError("JSON must contain 'Blocks' key")
    
    blocks = data['Blocks']
    if not blocks:
        raise ValueError("No blocks found in JSON")
    
    rows = []
    
    for block in blocks:
        try:
            # Only process LINE and WORD blocks that have text
            if block.get('BlockType') not in ['LINE', 'WORD']:
                continue
            
            if 'Text' not in block:
                continue
                
            text = block['Text']
            confidence = block.get('Confidence', 0)
            block_id = block.get('Id', '')
            block_type = block['BlockType']
            
            # Get geometry bounding box (normalized 0-1 coordinates)
            if 'Geometry' not in block or 'BoundingBox' not in block['Geometry']:
                continue
                
            bbox = block['Geometry']['BoundingBox']
            
            # Textract uses normalized coordinates (0-1)
            # Left = x position, Top = y position
            left = bbox.get('Left', 0)
            top = bbox.get('Top', 0)
            width = bbox.get('Width', 0)
            height = bbox.get('Height', 0)
            
            # Scale to canvas coordinates
            x1 = left * canvas_width
            y1 = top * canvas_height
            x2 = (left + width) * canvas_width
            y2 = (top + height) * canvas_height
            
            # Center coordinates
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            rows.append({
                'text': text,
                'block_type': block_type,
                'confidence': confidence,
                'block_id': block_id,
                'left_norm': left,
                'top_norm': top,
                'width_norm': width,
                'height_norm': height,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'center_x': center_x,
                'center_y': center_y,
                'width': x2 - x1,
                'height': y2 - y1
            })
            
        except (KeyError, ValueError, TypeError) as e:
            print(f"Error processing block: {e}")
            continue
    
    if not rows:
        raise ValueError("No valid text blocks could be parsed from JSON")
    
    return pd.DataFrame(rows)


def get_document_metadata(data: dict) -> Dict:
    """
    Extract document metadata from Textract JSON.
    
    Args:
        data: Dictionary from AWS Textract
        
    Returns:
        Dictionary with metadata
    """
    metadata = {}
    
    if 'DocumentMetadata' in data:
        metadata['pages'] = data['DocumentMetadata'].get('Pages', 1)
    else:
        metadata['pages'] = 1
    
    # Count different block types
    if 'Blocks' in data:
        blocks = data['Blocks']
        metadata['total_blocks'] = len(blocks)
        metadata['text_lines'] = sum(1 for b in blocks if b.get('BlockType') == 'LINE')
        metadata['words'] = sum(1 for b in blocks if b.get('BlockType') == 'WORD')
    
    return metadata


def scale_ocr_to_detections(ocr_df: pd.DataFrame, detection_canvas_width: float,
                           detection_canvas_height: float) -> pd.DataFrame:
    """
    Scale OCR coordinates to match detection canvas size.
    
    Args:
        ocr_df: DataFrame with OCR text blocks
        detection_canvas_width: Width of the detection coordinate space (without padding)
        detection_canvas_height: Height of the detection coordinate space (without padding)
        
    Returns:
        DataFrame with rescaled coordinates
    """
    df = ocr_df.copy()
    
    # Rescale using normalized coordinates
    df['x1'] = df['left_norm'] * detection_canvas_width
    df['y1'] = df['top_norm'] * detection_canvas_height
    df['x2'] = (df['left_norm'] + df['width_norm']) * detection_canvas_width
    df['y2'] = (df['top_norm'] + df['height_norm']) * detection_canvas_height
    
    df['center_x'] = (df['x1'] + df['x2']) / 2
    df['center_y'] = (df['y1'] + df['y2']) / 2
    df['width'] = df['x2'] - df['x1']
    df['height'] = df['y2'] - df['y1']
    
    return df

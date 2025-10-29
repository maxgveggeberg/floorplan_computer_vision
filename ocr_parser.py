import json
import pandas as pd
from typing import Dict, List, Tuple, Optional


def detect_ocr_format(data: dict) -> str:
    """
    Auto-detect whether the JSON is from AWS Textract or Google Cloud Vision.
    
    Args:
        data: Parsed JSON dictionary
        
    Returns:
        'textract' for AWS Textract, 'google_vision' for Google Cloud Vision
    """
    # Check for AWS Textract format
    if 'Blocks' in data and isinstance(data.get('Blocks'), list):
        # Additional check for Textract-specific fields
        if data['Blocks'] and 'BlockType' in data['Blocks'][0]:
            return 'textract'
    
    # Check for Google Cloud Vision format
    if 'fullTextAnnotation' in data:
        if 'pages' in data['fullTextAnnotation']:
            return 'google_vision'
    if 'textAnnotations' in data and isinstance(data.get('textAnnotations'), list):
        if data['textAnnotations']:
            return 'google_vision'
    
    # Default to textract for backward compatibility
    return 'textract'


def load_ocr_json(raw_bytes_or_str) -> dict:
    """
    Load OCR JSON from bytes or string (supports both AWS Textract and Google Cloud Vision).
    
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


# Keep the old function name for backward compatibility
def load_textract_json(raw_bytes_or_str) -> dict:
    """
    Load AWS Textract JSON from bytes or string.
    (Deprecated: Use load_ocr_json instead)
    
    Args:
        raw_bytes_or_str: Either bytes from file upload or string
        
    Returns:
        Parsed JSON as dictionary
    """
    return load_ocr_json(raw_bytes_or_str)


def parse_google_vision_blocks(data: dict, canvas_width: float = 1200, canvas_height: float = 900) -> pd.DataFrame:
    """
    Parse text blocks from Google Cloud Vision JSON into a pandas DataFrame.
    
    Args:
        data: Dictionary from Google Cloud Vision with 'fullTextAnnotation' key
        canvas_width: Width of the canvas to scale coordinates
        canvas_height: Height of the canvas to scale coordinates
    
    Returns:
        DataFrame with parsed text blocks including scaled coordinates
    """
    text_annotations = data.get('textAnnotations')
    if not text_annotations:
        raise ValueError("JSON must contain 'textAnnotations' key with annotations")

    # Skip the first entry because it contains the full text with a huge bounding box
    word_entries = text_annotations[1:] if len(text_annotations) > 1 else []

    if not word_entries:
        raise ValueError("No word-level annotations found in textAnnotations")

    # Determine image dimensions for normalization
    image_width = data.get('width')
    image_height = data.get('height')

    if not image_width or not image_height:
        max_x = 0
        max_y = 0
        for entry in word_entries:
            vertices = entry.get('boundingPoly', {}).get('vertices', [])
            for vertex in vertices:
                max_x = max(max_x, vertex.get('x', 0) or 0)
                max_y = max(max_y, vertex.get('y', 0) or 0)

        # Avoid division by zero
        image_width = image_width or max(max_x, 1)
        image_height = image_height or max(max_y, 1)

    rows = []

    for entry in word_entries:
        try:
            text = entry.get('description', '')
            if not text:
                continue

            vertices = entry.get('boundingPoly', {}).get('vertices', [])
            if len(vertices) < 4:
                continue

            x_coords = [v.get('x', 0) or 0 for v in vertices]
            y_coords = [v.get('y', 0) or 0 for v in vertices]

            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)

            if image_width == 0 or image_height == 0:
                continue

            left_norm = x_min / image_width
            top_norm = y_min / image_height
            width_norm = (x_max - x_min) / image_width
            height_norm = (y_max - y_min) / image_height

            x1 = left_norm * canvas_width
            y1 = top_norm * canvas_height
            x2 = (left_norm + width_norm) * canvas_width
            y2 = (top_norm + height_norm) * canvas_height

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            rows.append({
                'text': text,
                'block_type': 'WORD',
                'confidence': 100.0,
                'block_id': entry.get('mid', ''),
                'left_norm': left_norm,
                'top_norm': top_norm,
                'width_norm': width_norm,
                'height_norm': height_norm,
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
            print(f"Error processing text annotation: {e}")
            continue

    if not rows:
        raise ValueError("No valid text blocks could be parsed from Google Vision JSON")

    # Derive line-level entries by grouping words with significant vertical overlap
    word_rows = sorted(rows, key=lambda r: (r['top_norm'], r['left_norm']))
    line_rows: List[Dict] = []
    current_line: List[Dict] = []
    current_top: Optional[float] = None
    current_bottom: Optional[float] = None
    tolerance = 0.02

    def flush_line(words: List[Dict], line_index: int) -> None:
        if not words:
            return

        left_norm = min(w['left_norm'] for w in words)
        right_norm = max(w['left_norm'] + w['width_norm'] for w in words)
        top_norm = min(w['top_norm'] for w in words)
        bottom_norm = max(w['top_norm'] + w['height_norm'] for w in words)

        x1 = min(w['x1'] for w in words)
        y1 = min(w['y1'] for w in words)
        x2 = max(w['x2'] for w in words)
        y2 = max(w['y2'] for w in words)

        width = x2 - x1
        height = y2 - y1

        line_rows.append({
            'text': ' '.join(w['text'] for w in words),
            'block_type': 'LINE',
            'confidence': 100.0,
            'block_id': f"line_{line_index}",
            'left_norm': left_norm,
            'top_norm': top_norm,
            'width_norm': max(right_norm - left_norm, 0.0),
            'height_norm': max(bottom_norm - top_norm, 0.0),
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'center_x': (x1 + x2) / 2,
            'center_y': (y1 + y2) / 2,
            'width': width,
            'height': height
        })

    line_index = 0

    for word in word_rows:
        word_top = word['top_norm']
        word_bottom = word_top + word['height_norm']

        if not current_line:
            current_line = [word]
            current_top = word_top
            current_bottom = word_bottom
            continue

        assert current_top is not None and current_bottom is not None
        overlap = min(current_bottom, word_bottom) - max(current_top, word_top)
        min_height = min(current_bottom - current_top, word['height_norm'])

        if overlap >= max(min_height, 0) * 0.3 or word_top <= (current_bottom + tolerance):
            current_line.append(word)
            current_top = min(current_top, word_top)
            current_bottom = max(current_bottom, word_bottom)
        else:
            flush_line(current_line, line_index)
            line_index += 1
            current_line = [word]
            current_top = word_top
            current_bottom = word_bottom

    flush_line(current_line, line_index)

    rows.extend(line_rows)

    return pd.DataFrame(rows)


def parse_textract_blocks(data: dict, canvas_width: float = 1200, canvas_height: float = 900) -> pd.DataFrame:
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
        raise ValueError("No valid text blocks could be parsed from Textract JSON")
    
    return pd.DataFrame(rows)


def parse_text_blocks(data: dict, canvas_width: float = 1200, canvas_height: float = 900) -> pd.DataFrame:
    """
    Parse text blocks from OCR JSON into a pandas DataFrame.
    Auto-detects format (AWS Textract or Google Cloud Vision) and uses appropriate parser.
    
    Args:
        data: Dictionary from OCR service (AWS Textract or Google Cloud Vision)
        canvas_width: Width of the canvas to scale coordinates
        canvas_height: Height of the canvas to scale coordinates
    
    Returns:
        DataFrame with parsed text blocks including scaled coordinates
    """
    # Auto-detect format
    ocr_format = detect_ocr_format(data)
    
    if ocr_format == 'google_vision':
        return parse_google_vision_blocks(data, canvas_width, canvas_height)
    else:
        # Default to Textract parser for backward compatibility
        return parse_textract_blocks(data, canvas_width, canvas_height)


def get_document_metadata(data: dict) -> Dict:
    """
    Extract document metadata from OCR JSON (AWS Textract or Google Cloud Vision).
    
    Args:
        data: Dictionary from OCR service
        
    Returns:
        Dictionary with metadata
    """
    metadata = {}
    
    # Auto-detect format
    ocr_format = detect_ocr_format(data)
    
    if ocr_format == 'google_vision':
        # Handle Google Cloud Vision format
        if 'fullTextAnnotation' in data and 'pages' in data['fullTextAnnotation']:
            pages = data['fullTextAnnotation']['pages']
            metadata['pages'] = len(pages)

            total_blocks = 0
            total_words = 0
            total_paragraphs = 0
            
            for page in pages:
                blocks = page.get('blocks', [])
                total_blocks += len(blocks)
                
                for block in blocks:
                    paragraphs = block.get('paragraphs', [])
                    total_paragraphs += len(paragraphs)
                    
                    for paragraph in paragraphs:
                        words = paragraph.get('words', [])
                        total_words += len(words)
            
            metadata['total_blocks'] = total_blocks
            metadata['text_lines'] = total_paragraphs  # Map paragraphs to lines
            metadata['words'] = total_words
        elif 'textAnnotations' in data:
            annotations = data.get('textAnnotations') or []
            word_entries = annotations[1:] if len(annotations) > 1 else []
            metadata['pages'] = 1
            metadata['total_blocks'] = len(word_entries)
            metadata['text_lines'] = len(word_entries)
            metadata['words'] = len(word_entries)
        else:
            metadata['pages'] = 1
            metadata['total_blocks'] = 0
            metadata['text_lines'] = 0
            metadata['words'] = 0
        
        # Add image dimensions if available
        if 'width' in data:
            metadata['image_width'] = data['width']
        if 'height' in data:
            metadata['image_height'] = data['height']
    
    else:
        # Handle AWS Textract format
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
        else:
            metadata['total_blocks'] = 0
            metadata['text_lines'] = 0
            metadata['words'] = 0
    
    metadata['format'] = ocr_format
    
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

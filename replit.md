# Floorplan CV Viewer

## Overview

The Floorplan CV Viewer is a Streamlit-based web application designed to visualize and analyze object detection results from floor plan computer vision models. The application allows users to upload JSON files containing detection data (doors, windows, walls, etc.), persist them in a local SQLite database, and visualize the detections as interactive bounding boxes with filtering and summary capabilities.

The application also supports AWS Textract OCR results, enabling users to visualize text blocks extracted from floor plan images alongside the object detections.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology Choice: Streamlit**

The application uses Streamlit as its web framework, providing a simple, Python-native approach to building interactive web applications without requiring separate HTML/CSS/JavaScript development.

**Rationale:**
- Rapid development with minimal boilerplate code
- Built-in state management through `st.session_state`
- Native support for file uploads, forms, and interactive widgets
- Ideal for data science and ML visualization applications

**Key Components:**
- Main application (`app.py`): Orchestrates the UI, file uploads, and visualization
- Session state management for storing uploaded data, run IDs, and parsed DataFrames
- Sidebar controls for file uploads and configuration options
- Wide layout configuration for better visualization of floor plans

### Data Processing Layer

**CV Detection Parser (`cv_parser.py`)**

Handles parsing of object detection JSON files from YOLO-style computer vision models.

**Key Design Decision: Center vs. Top-Left Coordinate System**
- Default assumption: x,y coordinates represent box centers (YOLO-style)
- Computes corners as: `x1 = x - width/2`, `y1 = y - height/2`, `x2 = x + width/2`, `y2 = y + height/2`
- Provides flexibility through `assume_center` parameter to support alternative coordinate systems

**OCR Parser (`ocr_parser.py`)**

Handles parsing of AWS Textract JSON output for text block detection.

**Key Features:**
- Extracts LINE and WORD blocks from Textract responses
- Normalizes coordinates to canvas dimensions for consistent visualization
- Filters blocks to only include those with text content

**Visualization Engine (`viz.py`)**

Creates interactive Plotly visualizations for bounding box overlays.

**Design Decisions:**
- Deterministic color mapping for object classes (doors, windows, walls)
- Fallback color palette for unknown classes
- Configurable canvas dimensions to match different floor plan sizes

### Data Persistence Layer

**Technology Choice: SQLModel + SQLite**

The application uses SQLModel (built on SQLAlchemy and Pydantic) with SQLite for data persistence.

**Rationale:**
- SQLModel provides a clean, type-safe ORM layer
- SQLite requires no external database server (perfect for local applications)
- Pydantic integration ensures data validation
- Easy migration path to PostgreSQL if needed in the future

**Database Schema:**

1. **Run Table**: Records each uploaded CV detection JSON file
   - `id`: Primary key
   - `created_at`: Timestamp
   - `source_name`: File identifier
   - `raw_json`: Complete JSON for reproducibility

2. **Detection Table**: Individual detection records with foreign key to Run
   - Stores both center coordinates (x, y) and computed corners (x1, y1, x2, y2)
   - Includes confidence scores, class information, and unique detection IDs

3. **OCRRun Table**: Records each uploaded OCR JSON file
   - Similar structure to Run table
   - Additional `pages` field for multi-page documents

4. **OCRTextBlock Table**: Individual text block records with foreign key to OCRRun
   - Stores normalized coordinates and text content
   - Includes block type (LINE/WORD) and confidence scores

**Design Rationale:**
- Separate tables for runs vs. individual detections enables historical tracking
- Storing raw JSON alongside parsed data ensures data integrity and audit trail
- Normalized coordinates in separate columns optimize query performance for filtering

### Coordinate System Handling

**Problem:** Different CV models may output coordinates in different formats.

**Solution:** 
- Default interpretation treats (x, y) as box centers (YOLO convention)
- Mathematical transformation to compute corners: `x1, y1, x2, y2`
- UI toggle planned for alternative "top-left origin" interpretation

**Benefits:**
- Flexibility to handle multiple coordinate conventions
- Consistent internal representation for visualization
- Clear documentation of assumptions

### Visualization Strategy

**Technology Choice: Plotly**

Uses Plotly for interactive visualizations instead of static matplotlib plots.

**Advantages:**
- Interactive pan, zoom, and hover capabilities
- Easy integration with Streamlit
- Support for complex overlays and annotations
- Export capabilities for sharing results

**Color Mapping Strategy:**
- Predefined colors for common classes (door, window, wall)
- Deterministic fallback colors for unknown classes
- Consistent color assignment across sessions

## External Dependencies

### Core Framework
- **Streamlit**: Web application framework for Python data apps
- **Pandas**: Data manipulation and tabular display
- **Plotly**: Interactive visualization library

### Database
- **SQLModel**: ORM layer combining SQLAlchemy and Pydantic
- **SQLAlchemy**: SQL toolkit and ORM
- **SQLite**: Embedded relational database (no external service required)

### AWS Integration
- **AWS Textract**: OCR service for extracting text from floor plan images
  - Input format: JSON response from Textract's AnalyzeDocument API
  - Contains normalized bounding boxes and text content
  - Supports multi-page documents

### Data Formats
- **Input**: JSON files with specific schemas
  - CV Detection format: YOLO-style predictions with center coordinates
  - OCR format: AWS Textract Blocks with normalized bounding boxes
- **Storage**: SQLite database with relational schema

### Development Tools
- **Python 3.11+**: Runtime environment
- **JSON**: Data interchange format for CV model outputs
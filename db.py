from sqlmodel import SQLModel, Field, create_engine, Session, select
from typing import Optional, List
from datetime import datetime
import pandas as pd


class Run(SQLModel, table=True):
    """Record for each uploaded JSON file."""
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source_name: str
    raw_json: str


class Detection(SQLModel, table=True):
    """Individual detection record."""
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id")
    x: float
    y: float
    width: float
    height: float
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str
    class_id: int
    detection_uuid: str


class OCRRun(SQLModel, table=True):
    """Record for each uploaded OCR JSON file."""
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source_name: str
    raw_json: str
    pages: int = 1


class OCRTextBlock(SQLModel, table=True):
    """Individual OCR text block record."""
    id: Optional[int] = Field(default=None, primary_key=True)
    ocr_run_id: int = Field(foreign_key="ocrrun.id")
    text: str
    block_type: str
    confidence: float
    block_id: str
    left_norm: float
    top_norm: float
    width_norm: float
    height_norm: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float


engine = None


def init_db(db_path: str = "cv_viewer.db"):
    """
    Initialize the database and create tables.
    
    Args:
        db_path: Path to SQLite database file
    """
    global engine
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)


def create_run(source_name: str, raw_json: str) -> Run:
    """
    Create a new run record.
    
    Args:
        source_name: Name of the source file
        raw_json: Raw JSON content as string
        
    Returns:
        Created Run object
    """
    if engine is None:
        init_db()
    
    run = Run(source_name=source_name, raw_json=raw_json)
    
    with Session(engine) as session:
        session.add(run)
        session.commit()
        session.refresh(run)
        return run


def bulk_insert_detections(run_id: int, df: pd.DataFrame) -> int:
    """
    Bulk insert detections from DataFrame.
    
    Args:
        run_id: Foreign key to Run
        df: DataFrame with detection data
        
    Returns:
        Number of rows inserted
    """
    if engine is None:
        init_db()
    
    detections = []
    for _, row in df.iterrows():
        detection = Detection(
            run_id=run_id,
            x=row['x'],
            y=row['y'],
            width=row['width'],
            height=row['height'],
            x1=row['x1'],
            y1=row['y1'],
            x2=row['x2'],
            y2=row['y2'],
            confidence=row['confidence'],
            class_name=row['class_name'],
            class_id=row['class_id'],
            detection_uuid=row['detection_uuid']
        )
        detections.append(detection)
    
    with Session(engine) as session:
        session.add_all(detections)
        session.commit()
    
    return len(detections)


def list_runs() -> List[Run]:
    """
    List all runs in the database.
    
    Returns:
        List of Run objects
    """
    if engine is None:
        init_db()
    
    with Session(engine) as session:
        statement = select(Run).order_by(Run.created_at.desc())
        results = session.exec(statement)
        return list(results)


def load_detections(run_id: int) -> pd.DataFrame:
    """
    Load detections for a specific run.
    
    Args:
        run_id: Run ID to load detections for
        
    Returns:
        DataFrame with detection data
    """
    if engine is None:
        init_db()
    
    with Session(engine) as session:
        statement = select(Detection).where(Detection.run_id == run_id)
        results = session.exec(statement)
        detections = list(results)
        
        if not detections:
            return pd.DataFrame()
        
        data = []
        for det in detections:
            data.append({
                'x': det.x,
                'y': det.y,
                'width': det.width,
                'height': det.height,
                'x1': det.x1,
                'y1': det.y1,
                'x2': det.x2,
                'y2': det.y2,
                'confidence': det.confidence,
                'class_name': det.class_name,
                'class_id': det.class_id,
                'detection_uuid': det.detection_uuid,
                'area': det.width * det.height
            })
        
        return pd.DataFrame(data)


def create_ocr_run(source_name: str, raw_json: str, pages: int = 1) -> OCRRun:
    """
    Create a new OCR run record.
    
    Args:
        source_name: Name of the source file
        raw_json: Raw JSON content as string
        pages: Number of pages in document
        
    Returns:
        Created OCRRun object
    """
    if engine is None:
        init_db()
    
    ocr_run = OCRRun(source_name=source_name, raw_json=raw_json, pages=pages)
    
    with Session(engine) as session:
        session.add(ocr_run)
        session.commit()
        session.refresh(ocr_run)
        return ocr_run


def bulk_insert_ocr_blocks(ocr_run_id: int, df: pd.DataFrame) -> int:
    """
    Bulk insert OCR text blocks from DataFrame.
    
    Args:
        ocr_run_id: Foreign key to OCRRun
        df: DataFrame with OCR text block data
        
    Returns:
        Number of rows inserted
    """
    if engine is None:
        init_db()
    
    text_blocks = []
    for _, row in df.iterrows():
        block = OCRTextBlock(
            ocr_run_id=ocr_run_id,
            text=row['text'],
            block_type=row['block_type'],
            confidence=row['confidence'],
            block_id=row['block_id'],
            left_norm=row['left_norm'],
            top_norm=row['top_norm'],
            width_norm=row['width_norm'],
            height_norm=row['height_norm'],
            x1=row['x1'],
            y1=row['y1'],
            x2=row['x2'],
            y2=row['y2'],
            center_x=row['center_x'],
            center_y=row['center_y']
        )
        text_blocks.append(block)
    
    with Session(engine) as session:
        session.add_all(text_blocks)
        session.commit()
    
    return len(text_blocks)


def list_ocr_runs() -> List[OCRRun]:
    """
    List all OCR runs in the database.
    
    Returns:
        List of OCRRun objects
    """
    if engine is None:
        init_db()
    
    with Session(engine) as session:
        statement = select(OCRRun).order_by(OCRRun.created_at.desc())
        results = session.exec(statement)
        return list(results)


def load_ocr_blocks(ocr_run_id: int) -> pd.DataFrame:
    """
    Load OCR text blocks for a specific run.
    
    Args:
        ocr_run_id: OCR Run ID to load blocks for
        
    Returns:
        DataFrame with OCR text block data
    """
    if engine is None:
        init_db()
    
    with Session(engine) as session:
        statement = select(OCRTextBlock).where(OCRTextBlock.ocr_run_id == ocr_run_id)
        results = session.exec(statement)
        blocks = list(results)
        
        if not blocks:
            return pd.DataFrame()
        
        data = []
        for block in blocks:
            data.append({
                'text': block.text,
                'block_type': block.block_type,
                'confidence': block.confidence,
                'block_id': block.block_id,
                'left_norm': block.left_norm,
                'top_norm': block.top_norm,
                'width_norm': block.width_norm,
                'height_norm': block.height_norm,
                'x1': block.x1,
                'y1': block.y1,
                'x2': block.x2,
                'y2': block.y2,
                'center_x': block.center_x,
                'center_y': block.center_y,
                'width': block.x2 - block.x1,
                'height': block.y2 - block.y1
            })
        
        return pd.DataFrame(data)

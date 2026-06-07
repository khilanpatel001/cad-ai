"""
Engineering Drawing → Manufacturing Planning AI System
========================================================
Uses VLM (Claude Vision) + Traditional ML (scikit-learn) for feature recognition.
Run: python manufacturing_ai.py your_drawing.pdf
"""

import os, sys, json, base64, re, warnings, logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from pdf2image import convert_from_path
from pydantic import BaseModel, Field
import anthropic
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import networkx as nx

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("MFG-AI")

# ─────────────────────────── PYDANTIC MODELS ───────────────────────────

class TitleBlock(BaseModel):
    drawing_number: str = "N/A"
    part_number: str = "N/A"
    part_name: str = "N/A"
    material: str = "N/A"
    weight: str = "N/A"
    units: str = "mm"
    scale: str = "N/A"
    revision: str = "N/A"
    finish: str = "N/A"
    heat_treatment: str = "N/A"
    author: str = "N/A"
    approval: str = "N/A"
    drawing_date: str = "N/A"
    company: str = "N/A"

class Feature(BaseModel):
    feature_id: str
    feature_type: str
    description: str
    dimensions: dict = Field(default_factory=dict)
    tolerance: str = "N/A"
    gdt_callout: str = "N/A"
    surface_finish: str = "N/A"
    quantity: int = 1
    confidence: float = 0.0
    manufacturing_difficulty: str = "Medium"   # Low / Medium / High / Very High
    recommended_process: str = "N/A"
    source: str = "N/A"                        # which view/section it came from

class BOMItem(BaseModel):
    item_no: int
    part_number: str
    description: str
    material: str
    quantity: int
    weight: str = "N/A"
    item_type: str = "Manufactured"            # Manufactured / Purchased
    unit_cost: float = 0.0

class Operation(BaseModel):
    op_no: int
    operation_name: str
    machine: str
    tools: list[str]
    description: str
    setup_time_min: float = 30.0
    cycle_time_min: float = 0.0
    features_covered: list[str] = Field(default_factory=list)
    fixture: str = "N/A"
    coolant: str = "N/A"
    cutting_parameters: dict = Field(default_factory=dict)

class InspectionItem(BaseModel):
    item_no: int
    dimension: str
    nominal: str
    tolerance: str
    measurement_tool: str
    frequency: str = "100%"
    cmm_required: bool = False

class DFMRisk(BaseModel):
    risk_id: str
    category: str                              # Tolerance / Material / Process / Design
    description: str
    severity: str                              # Low / Medium / High / Critical
    recommendation: str

class ManufacturingPlan(BaseModel):
    title_block: TitleBlock
    features: list[Feature]
    bom: list[BOMItem]
    operations: list[Operation]
    inspection_plan: list[InspectionItem]
    dfm_risks: list[DFMRisk] = Field(default_factory=list)
    raw_material: dict = Field(default_factory=dict)
    cost_estimate: dict = Field(default_factory=dict)
    assembly_tree: dict = Field(default_factory=dict)
    cnc_summary: dict = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    drawing_inconsistencies: list[str] = Field(default_factory=list)
    analysis_confidence: float = 0.0


# ─────────────────────── STAGE 1: PDF INGESTION ───────────────────────

class DrawingIngester:
    """Converts PDF pages to high-res images for analysis."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def ingest(self, pdf_path: str) -> list[np.ndarray]:
        log.info(f"📄 Ingesting PDF: {pdf_path} @ {self.dpi} DPI")
        pil_images = convert_from_path(pdf_path, dpi=self.dpi)
        cv_images = []
        for i, pil_img in enumerate(pil_images):
            arr = np.array(pil_img.convert("RGB"))
            cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv_images.append(cv_img)
            log.info(f"  Page {i+1}: {cv_img.shape[1]}×{cv_img.shape[0]} px")
        return cv_images


# ─────────────── STAGE 2: TRADITIONAL ML FEATURE RECOGNITION ──────────

class TraditionalMLRecognizer:
    """
    scikit-learn based geometric feature detection using:
    - DBSCAN for hole/circle clustering
    - KNN classifier for shape pattern recognition
    - OpenCV contour analysis
    """

    def __init__(self):
        self._train_knn()

    def _train_knn(self):
        """Train KNN on synthetic geometric feature signatures."""
        X_train = np.array([
            # Circles / Holes
            [1.0,  0.95, 0.78, 0.98, 0.10],
            [1.0,  0.92, 0.75, 0.96, 0.08],
            [1.0,  0.90, 0.80, 0.97, 0.15],
            # Rectangles / Pockets / Slots
            [2.5,  0.55, 0.88, 0.94, 0.20],
            [3.5,  0.40, 0.90, 0.95, 0.18],
            [1.1,  0.75, 0.85, 0.92, 0.25],
            # Triangular/Angular
            [1.2,  0.45, 0.60, 0.70, 0.12],
            [1.3,  0.50, 0.55, 0.65, 0.09],
            # Long slots / Keyways
            [5.0,  0.25, 0.88, 0.96, 0.05],
            [6.0,  0.20, 0.90, 0.97, 0.04],
            # Complex shapes
            [1.5,  0.60, 0.70, 0.80, 0.30],
            [2.0,  0.65, 0.72, 0.82, 0.28],
        ])
        y_train = [
            "Hole", "Hole", "Hole",
            "Pocket", "Slot", "Pocket",
            "Angular_Feature", "Angular_Feature",
            "Keyway", "Keyway",
            "Complex_Profile", "Complex_Profile",
        ]
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(X_train, y_train)
        log.info("✅ KNN classifier trained on geometric feature signatures")

    def _compute_shape_descriptor(self, contour) -> Optional[np.ndarray]:
        area = cv2.contourArea(contour)
        if area < 100:
            return None
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / max(h, 1)
        rect_area = w * h
        extent = float(area) / max(rect_area, 1)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / max(hull_area, 1)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / max(perimeter ** 2, 1e-6)
        area_norm = area / 1e6
        return np.array([aspect_ratio, circularity, extent, solidity, area_norm])

    def detect_circles_dbscan(self, gray: np.ndarray) -> list[dict]:
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=50, param2=30, minRadius=5, maxRadius=200
        )
        detected = []
        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            if len(circles) > 1:
                coords = circles[:, :2].astype(float)
                scaler = StandardScaler()
                coords_scaled = scaler.fit_transform(coords)
                db = DBSCAN(eps=0.5, min_samples=1).fit(coords_scaled)
                labels = db.labels_
            else:
                labels = [0] * len(circles)

            for i, (cx, cy, r) in enumerate(circles):
                detected.append({
                    "center": (int(cx), int(cy)),
                    "radius_px": int(r),
                    "cluster": int(labels[i]),
                    "diameter_mm_est": round(r * 0.08, 2),
                })
        return detected

    def detect_contour_features(self, gray: np.ndarray) -> list[dict]:
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            desc = self._compute_shape_descriptor(cnt)
            if desc is None:
                continue
            pred = self.knn.predict([desc])[0]
            proba = self.knn.predict_proba([desc])[0].max()
            x, y, w, h = cv2.boundingRect(cnt)
            results.append({
                "feature_type": pred,
                "bbox": (x, y, w, h),
                "area_px2": int(area),
                "confidence": round(float(proba), 2),
                "width_mm_est": round(w * 0.08, 2),
                "height_mm_est": round(h * 0.08, 2),
            })
        results.sort(key=lambda r: r["area_px2"], reverse=True)
        return results[:20]

    def analyze(self, cv_image: np.ndarray) -> dict:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        circles = self.detect_circles_dbscan(gray)
        contours = self.detect_contour_features(gray)
        log.info(f"  ML: {len(circles)} circles, {len(contours)} contour features detected")
        return {"circles": circles, "contour_features": contours}


# ─────────────────── STAGE 3: VLM ANALYSIS (CLAUDE) ──────────────────

class VLMAnalyzer:
    """Uses Claude Vision to extract all drawing information with high accuracy."""

    # ── System prompt: full senior manufacturing engineer persona from doc 2 ──
    SYSTEM_PROMPT = """You are a world-class Manufacturing Engineering AI, Senior Process Planner,
Tooling Engineer, CNC Programmer, Cost Estimator, and BOM Specialist with 30+ years of experience
in aerospace, automotive, medical, and precision machining industries.

CRITICAL RULES:
1. NEVER hallucinate. Only extract information explicitly visible in the drawing.
2. NEVER assume dimensions not shown. Use "N/A" if not visible.
3. Every extracted value must include a confidence score (0.0 to 1.0).
4. If information is uncertain, mark source as "UNVERIFIED".
5. Cross-check all views: title block, dimension views, section views, detail views, BOM, notes.
6. Detect and report drawing inconsistencies (conflicting dimensions, missing callouts, etc.).
7. Validate manufacturability and flag DFM risks.
8. Flag missing dimensions, impossible tolerances, and quality risks.
9. Respond ONLY with valid JSON — no markdown, no explanations, no comments."""

    # ── Per-page extraction prompt ──
    PAGE_PROMPT_TEMPLATE = """Analyze engineering drawing page {page_num} of {total_pages}.

Traditional ML pre-scan detected these geometric hints (use as cross-check only, not gospel):
{ml_summary}

Extract EVERY visible detail from this page. Return a JSON object with this EXACT schema:

{{
  "page": {page_num},
  "title_block": {{
    "drawing_number":  {{"value": "string", "confidence": 0.0, "source": "title block"}},
    "part_number":     {{"value": "string", "confidence": 0.0, "source": "title block"}},
    "part_name":       {{"value": "string", "confidence": 0.0, "source": "title block"}},
    "material":        {{"value": "string", "confidence": 0.0, "source": "title block"}},
    "weight":          {{"value": "string", "confidence": 0.0, "source": "title block"}},
    "units":           {{"value": "mm or inch", "confidence": 0.0, "source": "title block"}},
    "scale":           {{"value": "string", "confidence": 0.0, "source": "title block"}},
    "revision":        {{"value": "string", "confidence": 0.0, "source": "title block"}},
    "finish":          {{"value": "string", "confidence": 0.0, "source": "title block"}},
    "heat_treatment":  {{"value": "string", "confidence": 0.0, "source": "title block"}},
    "author":          {{"value": "string", "confidence": 0.0, "source": "title block"}},
    "approval":        {{"value": "string", "confidence": 0.0, "source": "title block"}},
    "drawing_date":    {{"value": "string", "confidence": 0.0, "source": "title block"}},
    "company":         {{"value": "string", "confidence": 0.0, "source": "title block"}}
  }},
  "overall_dimensions": {{
    "length": {{"value": "string", "confidence": 0.0, "source": "view name"}},
    "width":  {{"value": "string", "confidence": 0.0, "source": "view name"}},
    "height": {{"value": "string", "confidence": 0.0, "source": "view name"}}
  }},
  "drawing_type": "Detail|Assembly|Section|Exploded",
  "complexity": "Low|Medium|High",
  "manufacturing_method": ["CNC Turning", "CNC Milling", "..."],
  "features": [
    {{
      "feature_id": "F001",
      "feature_type": "Hole|Thread|Pocket|Slot|Shaft|Keyway|Boss|Counterbore|Countersink|Chamfer|Fillet|Groove|Flange|Spline|Weld|Surface|Other",
      "description": "detailed description of what you see",
      "dimensions": {{
        "diameter":  {{"value": "string", "confidence": 0.0}},
        "depth":     {{"value": "string", "confidence": 0.0}},
        "length":    {{"value": "string", "confidence": 0.0}},
        "width":     {{"value": "string", "confidence": 0.0}},
        "angle":     {{"value": "string", "confidence": 0.0}},
        "radius":    {{"value": "string", "confidence": 0.0}},
        "thread":    {{"value": "string", "confidence": 0.0}},
        "pitch":     {{"value": "string", "confidence": 0.0}}
      }},
      "tolerance":              {{"value": "e.g. H7/g6 or ±0.05", "confidence": 0.0, "source": "dimension view"}},
      "gdt_callout":            {{"value": "GD&T symbols and values", "confidence": 0.0, "source": "view name"}},
      "surface_finish":         {{"value": "Ra value", "confidence": 0.0, "source": "view name"}},
      "quantity":               1,
      "blind_or_through":       "Blind|Through|N/A",
      "manufacturing_difficulty": "Low|Medium|High|Very High",
      "recommended_process":    "string",
      "source":                 "Front view|Top view|Section A-A|Detail B|etc",
      "confidence":             0.0
    }}
  ],
  "bom": [
    {{
      "item_no":     1,
      "part_number": "string",
      "description": "string",
      "material":    "string",
      "quantity":    1,
      "weight":      "string",
      "item_type":   "Manufactured|Purchased",
      "standard_hardware": false
    }}
  ],
  "drawing_inconsistencies": [
    "Description of any conflicting dimensions, missing callouts, or drawing errors found"
  ],
  "dfm_risks": [
    {{
      "risk_id":       "R001",
      "category":      "Tolerance|Material|Process|Design|Geometry",
      "description":   "description of the risk",
      "severity":      "Low|Medium|High|Critical",
      "recommendation": "suggested fix or mitigation"
    }}
  ],
  "cnc_summary": {{
    "setup_count":          1,
    "estimated_tool_changes": 0,
    "estimated_cycle_time_min": 0,
    "critical_operations": ["list of critical ops"],
    "recommended_cutting_params": {{
      "material": "string",
      "spindle_rpm": "string",
      "feed_rate": "string",
      "depth_of_cut": "string",
      "coolant": "Flood|Mist|Dry|Through-Spindle"
    }}
  }},
  "notes": ["any general notes, surface finish notes, heat treatment notes visible in drawing"],
  "confidence_overall": 0.0
}}

Important rules:
- Use "N/A" with confidence 0.0 for any field not visible in the drawing.
- Set confidence based on how clearly readable the value is (1.0 = crystal clear, 0.5 = partially legible, 0.1 = barely visible).
- List EVERY feature — holes, threads, pockets, slots, bosses, chamfers, fillets, grooves, weld symbols, surface finish symbols.
- For each dimension include the exact value as printed (e.g. "Ø12.5", "M10x1.5-6H", "R3", "45°").
- Identify standard purchased hardware (bolts, nuts, bearings, seals) and mark item_type="Purchased".
- Flag impossible tolerances (e.g. ±0.001mm on a rough cast surface) as DFM risks.
- Detect missing dimensions (e.g. a pocket with no depth callout) and list in drawing_inconsistencies."""

    # ── Synthesis prompt: merge multi-page results into one coherent plan ──
    SYNTHESIS_PROMPT = """You have analyzed {total_pages} pages of an engineering drawing.
Below are the per-page extraction results in JSON:

{pages_json}

Synthesize these into one unified JSON manufacturing plan. Merge features across pages,
resolve any conflicts by choosing the higher-confidence value, and produce a single object:

{{
  "title_block": {{ ... }},          // best values from any page, plain strings (not nested value/confidence)
  "features": [ ... ],              // all unique features, deduplicated, plain strings for all dimension values
  "bom": [ ... ],                   // merged BOM
  "drawing_inconsistencies": [ ... ],
  "dfm_risks": [ ... ],
  "cnc_summary": {{ ... }},
  "overall_dimensions": {{ "length": "val", "width": "val", "height": "val" }},
  "drawing_type": "...",
  "complexity": "...",
  "manufacturing_method": [ ... ],
  "notes": [ ... ],
  "confidence_overall": 0.0
}}

Rules:
- For title_block fields: use plain strings (e.g. "material": "EN24"), not nested objects.
- For feature dimensions: use plain strings (e.g. "diameter": "Ø12.5mm"), not nested objects.
- Deduplicate features that appear in multiple views (same feature_id or clearly the same feature).
- Set confidence_overall as the average of all page confidence_overall values.
- Do NOT hallucinate. Only use information from the per-page results.
- Return ONLY valid JSON, no markdown, no explanation."""

    def __init__(self):
        self.client = anthropic.Anthropic()

    def _image_to_b64(self, cv_img: np.ndarray) -> str:
        _, buf = cv2.imencode(".png", cv_img)
        return base64.standard_b64encode(buf).decode("utf-8")

    def _analyze_single_page(
        self,
        cv_img: np.ndarray,
        page_num: int,
        total_pages: int,
        ml_hints: dict,
    ) -> dict:
        """Run VLM analysis on a single drawing page."""
        b64 = self._image_to_b64(cv_img)
        ml_summary = json.dumps(ml_hints, indent=2)

        prompt = self.PAGE_PROMPT_TEMPLATE.format(
            page_num=page_num,
            total_pages=total_pages,
            ml_summary=ml_summary,
        )

        response = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=6000,
            system=self.SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": b64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            log.warning(f"Page {page_num} JSON parse error: {e}")
            return {"page": page_num, "features": [], "notes": [], "confidence_overall": 0.2}

    def _synthesize_pages(self, page_results: list[dict]) -> dict:
        """Merge multi-page results into one coherent plan using a second LLM call."""
        if len(page_results) == 1:
            # Single page — flatten nested value/confidence fields directly
            return self._flatten_single_page(page_results[0])

        pages_json = json.dumps(page_results, indent=2)
        prompt = self.SYNTHESIS_PROMPT.format(
            total_pages=len(page_results),
            pages_json=pages_json,
        )

        response = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=8000,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            log.warning(f"Synthesis JSON parse error: {e}")
            return page_results[0]  # fallback to first page

    def _flatten_single_page(self, page_data: dict) -> dict:
        """
        Convert nested {value, confidence, source} fields from a single-page result
        into plain strings for downstream processing.
        """
        def extract_val(obj):
            if isinstance(obj, dict) and "value" in obj:
                return obj["value"]
            return obj

        tb_raw = page_data.get("title_block", {})
        flat_tb = {k: extract_val(v) for k, v in tb_raw.items()}

        features = []
        for f in page_data.get("features", []):
            flat_dims = {}
            for dim_k, dim_v in f.get("dimensions", {}).items():
                val = extract_val(dim_v)
                if val and val != "N/A":
                    flat_dims[dim_k] = val
            flat_f = dict(f)
            flat_f["dimensions"] = flat_dims
            flat_f["tolerance"] = extract_val(f.get("tolerance", "N/A"))
            flat_f["gdt_callout"] = extract_val(f.get("gdt_callout", "N/A"))
            flat_f["surface_finish"] = extract_val(f.get("surface_finish", "N/A"))
            features.append(flat_f)

        od_raw = page_data.get("overall_dimensions", {})
        flat_od = {k: extract_val(v) for k, v in od_raw.items()}

        return {
            "title_block": flat_tb,
            "features": features,
            "bom": page_data.get("bom", []),
            "drawing_inconsistencies": page_data.get("drawing_inconsistencies", []),
            "dfm_risks": page_data.get("dfm_risks", []),
            "cnc_summary": page_data.get("cnc_summary", {}),
            "overall_dimensions": flat_od,
            "drawing_type": page_data.get("drawing_type", "Detail"),
            "complexity": page_data.get("complexity", "Medium"),
            "manufacturing_method": page_data.get("manufacturing_method", []),
            "notes": page_data.get("notes", []),
            "confidence_overall": page_data.get("confidence_overall", 0.5),
        }

    def analyze(self, cv_images: list[np.ndarray], ml_hints: list[dict]) -> dict:
        """Analyze all drawing pages and synthesize into one plan."""
        log.info(f"🤖 VLM (Claude) analyzing {len(cv_images)} page(s)...")
        total = len(cv_images)
        page_results = []

        for i, (img, hints) in enumerate(zip(cv_images, ml_hints), start=1):
            log.info(f"  Analyzing page {i}/{total}...")
            result = self._analyze_single_page(img, i, total, hints)
            page_results.append(result)

        log.info("🔗 Synthesizing multi-page results...")
        merged = self._synthesize_pages(page_results)
        log.info("✅ VLM analysis complete")
        return merged

    def _fallback_parse(self, raw: str) -> dict:
        return {
            "title_block": {"part_name": "Unknown Part", "material": "N/A", "units": "mm"},
            "features": [],
            "bom": [],
            "drawing_inconsistencies": [],
            "dfm_risks": [],
            "notes": [f"VLM raw output (parse failed): {raw[:500]}"],
            "confidence_overall": 0.2,
        }


# ──────────────── STAGE 4: MANUFACTURING PROCESS GENERATOR ───────────

FEATURE_TO_OPERATIONS = {
    "Hole": [
        ("Center Drilling", "CNC Turning / VMC", ["ø2 Center Drill"], "Dry", {}),
        ("Drilling", "CNC Turning / Drill Press", ["HSS/Carbide Drill"], "Flood", {"rpm": "800-1200", "feed": "0.15-0.25 mm/rev"}),
        ("Reaming", "CNC Turning / VMC", ["Reamer H7"], "Flood", {"rpm": "200-400", "feed": "0.05-0.1 mm/rev"}),
    ],
    "Thread": [
        ("Center Drilling", "VMC", ["Center Drill"], "Dry", {}),
        ("Drilling (Tap Size)", "VMC / Drill Press", ["Tap Drill"], "Flood", {}),
        ("Tapping", "VMC", ["Thread Tap / Thread Mill"], "Flood", {"rpm": "200-500"}),
    ],
    "Pocket": [
        ("Face Milling", "VMC", ["Ø50 Face Mill"], "Flood", {"rpm": "1200-2000", "feed": "500-800 mm/min"}),
        ("Pocket Roughing", "VMC", ["Ø10 4-Flute End Mill"], "Flood", {"rpm": "4000-6000", "feed": "800-1200 mm/min", "doc": "3-5mm"}),
        ("Pocket Finishing", "VMC", ["Ø10 2-Flute End Mill"], "Flood", {"rpm": "6000-8000", "feed": "400-600 mm/min", "doc": "0.2-0.5mm"}),
    ],
    "Slot": [
        ("Slot Milling Rough", "VMC", ["Ø8 End Mill"], "Flood", {"rpm": "3000-5000"}),
        ("Slot Milling Finish", "VMC", ["Ø8 End Mill Fine"], "Flood", {"rpm": "5000-8000"}),
    ],
    "Keyway": [
        ("Keyway Milling", "VMC / Key Seater", ["Woodruff Cutter / End Mill"], "Flood", {}),
    ],
    "Shaft": [
        ("Rough Turning", "CNC Lathe", ["CNMG Insert"], "Flood", {"rpm": "600-1000", "feed": "0.3-0.5 mm/rev", "doc": "2-4mm"}),
        ("Finish Turning", "CNC Lathe", ["VCGT Insert"], "Flood", {"rpm": "1500-2500", "feed": "0.05-0.15 mm/rev", "doc": "0.2-0.5mm"}),
        ("Parting / Grooving", "CNC Lathe", ["Grooving Insert"], "Flood", {}),
        ("Cylindrical Grinding", "CNC Grinder", ["CBN Wheel"], "Flood", {}),
    ],
    "Counterbore": [
        ("Center Drilling", "VMC", ["Center Drill"], "Dry", {}),
        ("Drilling", "VMC", ["Carbide Drill"], "Flood", {}),
        ("Counterboring", "VMC", ["Counterbore Tool"], "Flood", {}),
    ],
    "Countersink": [
        ("Center Drilling", "VMC", ["Center Drill"], "Dry", {}),
        ("Drilling", "VMC", ["Carbide Drill"], "Flood", {}),
        ("Countersinking", "VMC", ["90° Countersink"], "Flood", {}),
    ],
    "Surface": [
        ("Surface Grinding", "Surface Grinder", ["Alumina Wheel"], "Flood", {}),
    ],
    "Chamfer": [
        ("Chamfering", "CNC Lathe / VMC", ["Chamfer Tool"], "Flood", {}),
    ],
    "Boss": [
        ("Rough Turning", "CNC Lathe", ["CNMG Insert"], "Flood", {"rpm": "600-1000"}),
        ("Finish Turning", "CNC Lathe", ["VCGT Insert"], "Flood", {"rpm": "1500-2500"}),
    ],
    "Groove": [
        ("Grooving", "CNC Lathe", ["Grooving Insert"], "Flood", {}),
    ],
    "Flange": [
        ("Face Milling", "VMC", ["Ø63 Face Mill"], "Flood", {}),
        ("Contour Milling", "VMC", ["Ø12 End Mill"], "Flood", {}),
    ],
    "Spline": [
        ("Spline Hobbing / Milling", "Hobbing Machine / VMC", ["Hob Cutter / End Mill"], "Flood", {}),
    ],
}

STANDARD_OPS = [
    (10, "Raw Material Issue", "Store / ERP", [], "Issue raw material per BOM", "N/A", "N/A", {}),
    (20, "Incoming Inspection", "QC Area", ["Vernier Caliper", "Tape Measure"], "Verify raw material dimensions & material cert", "N/A", "N/A", {}),
    (30, "Marking & Cutting", "Bandsaw / Cutting Machine", ["Bandsaw Blade"], "Cut to rough length per BOM", "N/A", "N/A", {}),
]

FINAL_OPS = [
    ("Deburring & Cleaning", "Bench / Deburring Machine", ["Deburring Tool", "File"], "Remove all burrs, clean with solvent", "N/A", "N/A", {}),
    ("Final Inspection", "CMM / QC Area", ["CMM", "Micrometer", "Bore Gauge"], "Inspect per inspection plan", "N/A", "N/A", {}),
    ("Surface Treatment", "Plating / Paint Shop", [], "As per drawing specification", "N/A", "N/A", {}),
    ("Packing & Dispatch", "Packing Area", [], "Pack per customer requirements; attach inspection report", "N/A", "N/A", {}),
]

TOOL_GAUGE_MAP = {
    "Hole":        ("Bore Gauge / Pin Gauge", False),
    "Thread":      ("Thread Gauge (Go/No-Go)", False),
    "Pocket":      ("Depth Gauge / CMM", True),
    "Slot":        ("Vernier Caliper / CMM", False),
    "Shaft":       ("Micrometer / CMM", True),
    "Surface":     ("Surface Roughness Tester (Profilometer)", False),
    "Keyway":      ("Vernier Caliper / Optical Comparator", False),
    "Counterbore": ("Depth Micrometer / CMM", True),
    "Countersink": ("Countersink Gauge", False),
    "Chamfer":     ("Optical Comparator / CMM", False),
    "Boss":        ("Micrometer", False),
    "Groove":      ("Groove Micrometer", False),
    "Flange":      ("CMM", True),
    "Spline":      ("Spline Gauge / CMM", True),
}

MATERIAL_COST = {
    "EN8": 85, "EN24": 120, "EN31": 150, "SS304": 220, "SS316": 280,
    "Al6061": 180, "Al7075": 240, "MS": 65, "Cast Iron": 75,
    "Brass": 320, "Bronze": 380, "Titanium": 950, "Inconel": 1800,
    "default": 100,
}
MACHINE_RATE = {  # ₹/hr
    "CNC Lathe": 800, "VMC": 1200, "CNC Grinder": 900,
    "Surface Grinder": 600, "Drill Press": 300, "Bench": 200,
    "CMM": 1000, "Hobbing Machine": 1100, "default": 500,
}


class ManufacturingPlanGenerator:
    def generate(self, vlm_data: dict, ml_data: dict) -> ManufacturingPlan:
        log.info("⚙️ Generating manufacturing plan...")

        tb_raw = vlm_data.get("title_block", {})
        # Flatten any residual nested value/confidence dicts
        flat_tb = {}
        for k in TitleBlock.model_fields:
            v = tb_raw.get(k, "N/A")
            flat_tb[k] = v["value"] if isinstance(v, dict) and "value" in v else str(v)
        title_block = TitleBlock(**flat_tb)

        features = self._build_features(vlm_data, ml_data)
        operations = self._build_operations(features, title_block)
        bom = self._build_bom(vlm_data, title_block)
        inspection = self._build_inspection(features)
        dfm_risks = self._build_dfm_risks(vlm_data)
        raw_mat = self._calc_raw_material(vlm_data, title_block)
        cost = self._estimate_cost(operations, raw_mat, title_block)
        assembly_tree = self._build_assembly_tree(bom, title_block)
        cnc_summary = vlm_data.get("cnc_summary", {})

        return ManufacturingPlan(
            title_block=title_block,
            features=features,
            bom=bom,
            operations=operations,
            inspection_plan=inspection,
            dfm_risks=dfm_risks,
            raw_material=raw_mat,
            cost_estimate=cost,
            assembly_tree=assembly_tree,
            cnc_summary=cnc_summary,
            notes=vlm_data.get("notes", []),
            drawing_inconsistencies=vlm_data.get("drawing_inconsistencies", []),
            analysis_confidence=vlm_data.get("confidence_overall", 0.7),
        )

    def _extract_str(self, val) -> str:
        """Safely extract string from plain str or nested {value:...} dict."""
        if isinstance(val, dict) and "value" in val:
            return str(val["value"])
        return str(val) if val is not None else "N/A"

    def _build_features(self, vlm_data: dict, ml_data: dict) -> list[Feature]:
        features = []
        for i, f in enumerate(vlm_data.get("features", []), 1):
            # Flatten dimensions
            raw_dims = f.get("dimensions", {})
            flat_dims = {}
            for k, v in raw_dims.items():
                sv = self._extract_str(v)
                if sv and sv not in ("N/A", ""):
                    flat_dims[k] = sv

            features.append(Feature(
                feature_id=f.get("feature_id", f"F{i:03d}"),
                feature_type=f.get("feature_type", "Other"),
                description=f.get("description", "N/A"),
                dimensions=flat_dims,
                tolerance=self._extract_str(f.get("tolerance", "N/A")),
                gdt_callout=self._extract_str(f.get("gdt_callout", "N/A")),
                surface_finish=self._extract_str(f.get("surface_finish", "N/A")),
                quantity=int(f.get("quantity", 1)),
                confidence=float(f.get("confidence", 0.7)),
                manufacturing_difficulty=f.get("manufacturing_difficulty", "Medium"),
                recommended_process=f.get("recommended_process", "N/A"),
                source=f.get("source", "N/A"),
            ))

        # Supplement with ML-detected circles if VLM missed them
        ml_circles = ml_data.get("circles", [])
        for i, c in enumerate(ml_circles):
            fid = f"ML_HOLE_{i+1:03d}"
            if not any(fid in ff.feature_id for ff in features):
                features.append(Feature(
                    feature_id=fid,
                    feature_type="Hole",
                    description=f"ML-detected circle Ø~{c['diameter_mm_est']}mm (verify with drawing)",
                    dimensions={"diameter": f"~{c['diameter_mm_est']}mm (ML estimate)"},
                    confidence=0.45,
                    manufacturing_difficulty="Low",
                    source="ML pre-scan",
                ))
        log.info(f"  {len(features)} features compiled")
        return features

    def _build_operations(self, features: list[Feature], tb: TitleBlock) -> list[Operation]:
        ops = []
        op_no = 10
        for op_no_base, name, machine, tools, desc, fixture, coolant, params in STANDARD_OPS:
            ops.append(Operation(
                op_no=op_no_base, operation_name=name, machine=machine,
                tools=tools, description=desc, setup_time_min=15, cycle_time_min=10,
                fixture=fixture, coolant=coolant, cutting_parameters=params,
            ))

        op_no = 40
        seen_types = set()
        for feat in features:
            ftype = feat.feature_type
            if ftype in FEATURE_TO_OPERATIONS and ftype not in seen_types:
                seen_types.add(ftype)
                for sub_name, machine, tools, coolant, params in FEATURE_TO_OPERATIONS[ftype]:
                    ops.append(Operation(
                        op_no=op_no, operation_name=sub_name, machine=machine,
                        tools=tools,
                        description=f"{sub_name} for {ftype} feature(s) — {feat.description[:80]}",
                        setup_time_min=30,
                        cycle_time_min=max(5, feat.quantity * 3),
                        features_covered=[feat.feature_id],
                        fixture="Soft Jaw / Fixture Plate",
                        coolant=coolant,
                        cutting_parameters=params,
                    ))
                    op_no += 10

        for name, machine, tools, desc, fixture, coolant, params in FINAL_OPS:
            ops.append(Operation(
                op_no=op_no, operation_name=name, machine=machine,
                tools=tools, description=desc, setup_time_min=10, cycle_time_min=15,
                fixture=fixture, coolant=coolant, cutting_parameters=params,
            ))
            op_no += 10

        log.info(f"  {len(ops)} operations generated")
        return ops

    def _build_bom(self, vlm_data: dict, tb: TitleBlock) -> list[BOMItem]:
        bom = []
        raw_bom = vlm_data.get("bom", [])
        if raw_bom:
            for item in raw_bom:
                mat = item.get("material", tb.material)
                if isinstance(mat, dict):
                    mat = mat.get("value", tb.material)
                price = MATERIAL_COST.get(mat, MATERIAL_COST["default"])
                bom.append(BOMItem(
                    item_no=item.get("item_no", len(bom)+1),
                    part_number=item.get("part_number", f"PRT-{len(bom)+1:03d}"),
                    description=item.get("description", "Component"),
                    material=mat,
                    quantity=int(item.get("quantity", 1)),
                    weight=str(item.get("weight", "N/A")),
                    item_type=item.get("item_type", "Manufactured"),
                    unit_cost=round(price * 0.5, 2),
                ))
        else:
            bom.append(BOMItem(
                item_no=1,
                part_number=tb.part_number,
                description=tb.part_name,
                material=tb.material,
                quantity=1,
                weight=tb.weight,
                item_type="Manufactured",
                unit_cost=MATERIAL_COST.get(tb.material, MATERIAL_COST["default"]) * 0.5,
            ))
        log.info(f"  {len(bom)} BOM items compiled")
        return bom

    def _build_inspection(self, features: list[Feature]) -> list[InspectionItem]:
        items = []
        i = 1
        for feat in features:
            if feat.confidence < 0.4:
                continue
            gauge, cmm = TOOL_GAUGE_MAP.get(feat.feature_type, ("Vernier Caliper", False))
            for dim_name, dim_val in feat.dimensions.items():
                if dim_val and dim_val not in ("N/A", ""):
                    items.append(InspectionItem(
                        item_no=i,
                        dimension=f"{feat.feature_type} {dim_name}",
                        nominal=str(dim_val),
                        tolerance=feat.tolerance,
                        measurement_tool=gauge,
                        frequency="100%" if feat.manufacturing_difficulty in ("High", "Very High") else "First Off + 10%",
                        cmm_required=cmm,
                    ))
                    i += 1
            # Add GD&T callout as separate inspection row
            if feat.gdt_callout and feat.gdt_callout not in ("N/A", ""):
                items.append(InspectionItem(
                    item_no=i,
                    dimension=f"{feat.feature_type} GD&T",
                    nominal=feat.gdt_callout,
                    tolerance="Per GD&T callout",
                    measurement_tool="CMM",
                    frequency="First Article + 10%",
                    cmm_required=True,
                ))
                i += 1

        items.append(InspectionItem(
            item_no=i, dimension="Surface Finish (Ra)", nominal="Per Drawing",
            tolerance="N/A", measurement_tool="Profilometer / Ra Tester",
            frequency="First Off + 10%", cmm_required=False,
        ))
        log.info(f"  {len(items)} inspection checkpoints generated")
        return items

    def _build_dfm_risks(self, vlm_data: dict) -> list[DFMRisk]:
        risks = []
        for r in vlm_data.get("dfm_risks", []):
            risks.append(DFMRisk(
                risk_id=r.get("risk_id", f"R{len(risks)+1:03d}"),
                category=r.get("category", "Design"),
                description=r.get("description", "N/A"),
                severity=r.get("severity", "Medium"),
                recommendation=r.get("recommendation", "Review with design team"),
            ))
        log.info(f"  {len(risks)} DFM risks identified")
        return risks

    def _calc_raw_material(self, vlm_data: dict, tb: TitleBlock) -> dict:
        od = vlm_data.get("overall_dimensions", {})

        def safe_dim(key):
            v = od.get(key, "N/A")
            if isinstance(v, dict):
                v = v.get("value", "N/A")
            return str(v)

        length = safe_dim("length")
        width  = safe_dim("width")
        height = safe_dim("height")

        def add_allowance(dim_str, allowance=10):
            try:
                num = float(re.sub(r"[^0-9.]", "", str(dim_str)))
                return f"{num + allowance:.1f} mm"
            except Exception:
                return f"{dim_str} + {allowance}mm allowance"

        return {
            "material": tb.material,
            "finished_length": length,
            "finished_width": width,
            "finished_height": height,
            "raw_stock_length": add_allowance(length),
            "raw_stock_diameter": f"Ø{add_allowance(width, 5)}" if width != "N/A" else "N/A",
            "machining_allowance": "5mm per side",
            "material_utilization": "~75%",
            "estimated_scrap": "~25%",
        }

    def _estimate_cost(self, ops: list[Operation], raw_mat: dict, tb: TitleBlock) -> dict:
        mat_name = tb.material
        mat_rate = MATERIAL_COST.get(mat_name, MATERIAL_COST["default"])
        mat_cost = mat_rate * 1.2

        machining_cost = 0.0
        for op in ops:
            rate = next((v for k, v in MACHINE_RATE.items() if k in op.machine), MACHINE_RATE["default"])
            total_min = op.setup_time_min + op.cycle_time_min
            machining_cost += rate * total_min / 60

        tooling_cost = len(ops) * 50
        inspection_cost = 500
        overhead = (mat_cost + machining_cost) * 0.15
        total = mat_cost + machining_cost + tooling_cost + inspection_cost + overhead

        return {
            "material_cost": round(mat_cost, 2),
            "machining_cost": round(machining_cost, 2),
            "tooling_cost": round(tooling_cost, 2),
            "inspection_cost": round(inspection_cost, 2),
            "overhead": round(overhead, 2),
            "total_cost": round(total, 2),
            "currency": "INR",
            "note": "Estimate only — actual costs depend on batch quantity and supplier rates",
        }

    def _build_assembly_tree(self, bom: list[BOMItem], tb: TitleBlock) -> dict:
        G = nx.DiGraph()
        root = tb.part_number or "ASSEMBLY"
        G.add_node(root, description=tb.part_name)
        for item in bom:
            G.add_node(item.part_number, description=item.description)
            G.add_edge(root, item.part_number, qty=item.quantity)
        return {
            "root": root,
            "nodes": list(G.nodes(data=True)),
            "edges": [(u, v, d) for u, v, d in G.edges(data=True)],
        }


# ────────────────────── EXCEL REPORT GENERATOR ────────────────────────

class ExcelReportGenerator:
    HEADER_FILL  = PatternFill("solid", fgColor="1F3864")
    HEADER_FONT  = Font(color="FFFFFF", bold=True, name="Calibri", size=10)
    ALT_FILL     = PatternFill("solid", fgColor="E8EEF4")
    WARN_FILL    = PatternFill("solid", fgColor="FFF2CC")
    RISK_FILL    = PatternFill("solid", fgColor="FFE0E0")
    BORDER       = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin"),
    )

    def _header(self, ws, row, cols):
        for col_idx, col_name in enumerate(cols, 1):
            c = ws.cell(row=row, column=col_idx, value=col_name)
            c.fill = self.HEADER_FILL
            c.font = self.HEADER_FONT
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            c.border = self.BORDER

    def _row(self, ws, row, values, alt=False, warn=False, risk=False):
        fill = self.RISK_FILL if risk else (self.WARN_FILL if warn else (self.ALT_FILL if alt else PatternFill()))
        for col_idx, val in enumerate(values, 1):
            c = ws.cell(row=row, column=col_idx, value=val)
            c.fill = fill
            c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
            c.border = self.BORDER

    def generate(self, plan: ManufacturingPlan, output_path: str):
        wb = openpyxl.Workbook()
        wb.remove(wb.active)

        self._sheet_summary(wb, plan)
        self._sheet_bom(wb, plan)
        self._sheet_routing(wb, plan)
        self._sheet_inspection(wb, plan)
        self._sheet_features(wb, plan)
        self._sheet_dfm(wb, plan)
        self._sheet_cost(wb, plan)
        self._sheet_raw_material(wb, plan)
        self._sheet_cnc(wb, plan)

        wb.save(output_path)
        log.info(f"📊 Excel report saved: {output_path}")

    def _sheet_summary(self, wb, plan: ManufacturingPlan):
        ws = wb.create_sheet("SUMMARY")
        ws.column_dimensions["A"].width = 30
        ws.column_dimensions["B"].width = 42
        tb = plan.title_block

        title_font = Font(bold=True, size=16, color="1F3864", name="Calibri")
        ws["A1"] = "MANUFACTURING PLANNING PACKAGE"
        ws["A1"].font = title_font
        ws.merge_cells("A1:B1")
        ws["A1"].alignment = Alignment(horizontal="center")

        rows = [
            ("Drawing Number", tb.drawing_number),
            ("Part Number", tb.part_number),
            ("Part Name", tb.part_name),
            ("Material", tb.material),
            ("Weight", tb.weight),
            ("Units", tb.units),
            ("Scale", tb.scale),
            ("Revision", tb.revision),
            ("Finish", tb.finish),
            ("Heat Treatment", tb.heat_treatment),
            ("Author", tb.author),
            ("Approval", tb.approval),
            ("Date", tb.drawing_date),
            ("Company", tb.company),
            ("", ""),
            ("Total Features", len(plan.features)),
            ("Total BOM Items", len(plan.bom)),
            ("Total Operations", len(plan.operations)),
            ("Inspection Points", len(plan.inspection_plan)),
            ("DFM Risks Identified", len(plan.dfm_risks)),
            ("Drawing Inconsistencies", len(plan.drawing_inconsistencies)),
            ("Analysis Confidence", f"{plan.analysis_confidence*100:.1f}%"),
            ("Report Generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ]
        for i, (k, v) in enumerate(rows, 3):
            ws.cell(row=i, column=1, value=k).font = Font(bold=True, name="Calibri")
            ws.cell(row=i, column=2, value=str(v))

        # Drawing Inconsistencies
        if plan.drawing_inconsistencies:
            r = len(rows) + 5
            ws.cell(row=r, column=1, value="⚠ DRAWING INCONSISTENCIES").font = Font(bold=True, color="CC0000", name="Calibri")
            for note in plan.drawing_inconsistencies:
                r += 1
                c = ws.cell(row=r, column=1, value=f"• {note}")
                c.font = Font(name="Calibri", size=9, color="CC0000")
                c.fill = self.WARN_FILL
                ws.merge_cells(f"A{r}:B{r}")

        # Notes
        if plan.notes:
            r = len(rows) + 5 + len(plan.drawing_inconsistencies) + 3
            ws.cell(row=r, column=1, value="DRAWING NOTES").font = Font(bold=True, color="1F3864", name="Calibri")
            for note in plan.notes:
                r += 1
                ws.cell(row=r, column=1, value=f"• {note}").font = Font(name="Calibri", size=9)
                ws.merge_cells(f"A{r}:B{r}")

    def _sheet_bom(self, wb, plan: ManufacturingPlan):
        ws = wb.create_sheet("BILL OF MATERIALS")
        cols = ["Item No", "Part Number", "Description", "Material", "Qty", "Weight", "Type", "Unit Cost (INR)"]
        for w, col in zip([8, 18, 30, 15, 6, 12, 14, 16], range(1, 9)):
            ws.column_dimensions[get_column_letter(col)].width = w
        ws.row_dimensions[1].height = 30
        self._header(ws, 1, cols)
        for i, item in enumerate(plan.bom, 2):
            self._row(ws, i, [item.item_no, item.part_number, item.description, item.material,
                               item.quantity, item.weight, item.item_type, item.unit_cost], i % 2 == 0)

    def _sheet_routing(self, wb, plan: ManufacturingPlan):
        ws = wb.create_sheet("PROCESS ROUTING")
        cols = ["Op No", "Operation", "Machine / Work Centre", "Fixture", "Tools Required",
                "Description", "Coolant", "Setup Time (min)", "Cycle Time (min)"]
        for w, col in zip([8, 22, 25, 16, 30, 35, 10, 14, 14], range(1, 10)):
            ws.column_dimensions[get_column_letter(col)].width = w
        ws.row_dimensions[1].height = 30
        self._header(ws, 1, cols)
        for i, op in enumerate(plan.operations, 2):
            self._row(ws, i, [
                op.op_no, op.operation_name, op.machine, op.fixture,
                ", ".join(op.tools), op.description, op.coolant,
                op.setup_time_min, op.cycle_time_min,
            ], i % 2 == 0)

    def _sheet_inspection(self, wb, plan: ManufacturingPlan):
        ws = wb.create_sheet("INSPECTION PLAN")
        cols = ["Item", "Dimension / Characteristic", "Nominal Value",
                "Tolerance", "Measurement Tool", "CMM Required", "Frequency", "Result", "Status"]
        for w, col in zip([6, 28, 16, 14, 28, 12, 14, 14, 10], range(1, 10)):
            ws.column_dimensions[get_column_letter(col)].width = w
        ws.row_dimensions[1].height = 30
        self._header(ws, 1, cols)
        for i, item in enumerate(plan.inspection_plan, 2):
            self._row(ws, i, [
                item.item_no, item.dimension, item.nominal,
                item.tolerance, item.measurement_tool,
                "Yes" if item.cmm_required else "No",
                item.frequency, "", "",
            ], i % 2 == 0)

    def _sheet_features(self, wb, plan: ManufacturingPlan):
        ws = wb.create_sheet("FEATURES")
        cols = ["Feature ID", "Type", "Description", "Dimensions", "Tolerance",
                "GD&T", "Surface Finish", "Difficulty", "Process", "Source", "Qty", "Confidence"]
        for w, col in zip([12, 18, 32, 28, 14, 18, 14, 10, 20, 18, 6, 12], range(1, 13)):
            ws.column_dimensions[get_column_letter(col)].width = w
        ws.row_dimensions[1].height = 30
        self._header(ws, 1, cols)
        for i, f in enumerate(plan.features, 2):
            dim_str = ", ".join(f"{k}={v}" for k, v in f.dimensions.items() if v)
            risk = f.manufacturing_difficulty in ("High", "Very High")
            self._row(ws, i, [
                f.feature_id, f.feature_type, f.description, dim_str,
                f.tolerance, f.gdt_callout, f.surface_finish,
                f.manufacturing_difficulty, f.recommended_process,
                f.source, f.quantity, f"{f.confidence*100:.0f}%",
            ], i % 2 == 0, risk=risk)

    def _sheet_dfm(self, wb, plan: ManufacturingPlan):
        ws = wb.create_sheet("DFM RISKS")
        cols = ["Risk ID", "Category", "Description", "Severity", "Recommendation"]
        for w, col in zip([10, 16, 45, 12, 45], range(1, 6)):
            ws.column_dimensions[get_column_letter(col)].width = w
        ws.row_dimensions[1].height = 30
        self._header(ws, 1, cols)
        severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        sorted_risks = sorted(plan.dfm_risks, key=lambda r: severity_order.get(r.severity, 9))
        for i, r in enumerate(sorted_risks, 2):
            is_critical = r.severity in ("Critical", "High")
            self._row(ws, i, [
                r.risk_id, r.category, r.description, r.severity, r.recommendation,
            ], i % 2 == 0, risk=is_critical)

        if not plan.dfm_risks:
            ws.cell(row=2, column=1, value="No DFM risks identified.").font = Font(italic=True, name="Calibri")

    def _sheet_cost(self, wb, plan: ManufacturingPlan):
        ws = wb.create_sheet("COST ESTIMATE")
        ws.column_dimensions["A"].width = 30
        ws.column_dimensions["B"].width = 22
        cost = plan.cost_estimate
        self._header(ws, 1, ["Cost Element", "Amount (INR)"])
        items = [
            ("Material Cost", cost.get("material_cost", 0)),
            ("Machining Cost", cost.get("machining_cost", 0)),
            ("Tooling Cost", cost.get("tooling_cost", 0)),
            ("Inspection Cost", cost.get("inspection_cost", 0)),
            ("Overhead (15%)", cost.get("overhead", 0)),
        ]
        for i, (k, v) in enumerate(items, 2):
            self._row(ws, i, [k, f"₹ {v:,.2f}"], i % 2 == 0)
        total_row = len(items) + 2
        ws.cell(row=total_row, column=1, value="TOTAL COST").font = Font(bold=True, color="FF0000", name="Calibri")
        ws.cell(row=total_row, column=2, value=f"₹ {cost.get('total_cost', 0):,.2f}").font = Font(bold=True, color="FF0000")
        note_row = total_row + 2
        ws.cell(row=note_row, column=1, value=cost.get("note", ""))
        ws.merge_cells(f"A{note_row}:B{note_row}")

    def _sheet_raw_material(self, wb, plan: ManufacturingPlan):
        ws = wb.create_sheet("RAW MATERIAL")
        ws.column_dimensions["A"].width = 30
        ws.column_dimensions["B"].width = 32
        rm = plan.raw_material
        self._header(ws, 1, ["Parameter", "Value"])
        items = [(k.replace("_", " ").title(), str(v)) for k, v in rm.items()]
        for i, (k, v) in enumerate(items, 2):
            self._row(ws, i, [k, v], i % 2 == 0)

    def _sheet_cnc(self, wb, plan: ManufacturingPlan):
        ws = wb.create_sheet("CNC SUMMARY")
        ws.column_dimensions["A"].width = 35
        ws.column_dimensions["B"].width = 35
        cnc = plan.cnc_summary
        self._header(ws, 1, ["Parameter", "Value"])
        if cnc:
            flat = []
            for k, v in cnc.items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        flat.append((f"{k.replace('_',' ').title()} — {sk.replace('_',' ').title()}", str(sv)))
                elif isinstance(v, list):
                    flat.append((k.replace("_", " ").title(), ", ".join(str(x) for x in v)))
                else:
                    flat.append((k.replace("_", " ").title(), str(v)))
            for i, (k, v) in enumerate(flat, 2):
                self._row(ws, i, [k, v], i % 2 == 0)
        else:
            ws.cell(row=2, column=1, value="CNC summary not available for this drawing.").font = Font(italic=True, name="Calibri")


# ──────────────────────────── MAIN RUNNER ─────────────────────────────

def run(pdf_path: str):
    path = Path(pdf_path)
    if not path.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    print("\n" + "="*65)
    print("  ENGINEERING DRAWING → MANUFACTURING PLANNING AI SYSTEM")
    print("="*65)

    out_dir = Path("mfg_output")
    out_dir.mkdir(exist_ok=True)

    # Stage 1: Ingest
    ingester = DrawingIngester(dpi=300)
    cv_images = ingester.ingest(str(path))

    # Stage 2: Traditional ML (all pages)
    log.info("🔬 Running Traditional ML (scikit-learn) feature recognition...")
    ml_recognizer = TraditionalMLRecognizer()
    ml_results = [ml_recognizer.analyze(img) for img in cv_images]

    # Stage 3: VLM — all pages analyzed, then synthesized
    vlm = VLMAnalyzer()
    vlm_data = vlm.analyze(cv_images, ml_results)

    # Stage 4: Generate manufacturing plan
    generator = ManufacturingPlanGenerator()
    plan = generator.generate(vlm_data, ml_results[0] if ml_results else {})

    # Excel output
    excel_path = out_dir / f"{path.stem}_manufacturing_plan.xlsx"
    ExcelReportGenerator().generate(plan, str(excel_path))

    # JSON output
    json_path = out_dir / f"{path.stem}_manufacturing_plan.json"
    with open(json_path, "w") as f:
        json.dump(plan.model_dump(), f, indent=2, default=str)
    log.info(f"📋 JSON output saved: {json_path}")

    # Print summary
    print("\n" + "─"*65)
    print("  ANALYSIS SUMMARY")
    print("─"*65)
    tb = plan.title_block
    print(f"  Drawing No   : {tb.drawing_number}")
    print(f"  Part Name    : {tb.part_name}")
    print(f"  Material     : {tb.material}")
    print(f"  Revision     : {tb.revision}")
    print(f"  Pages        : {len(cv_images)}")
    print(f"  Features     : {len(plan.features)}")
    print(f"  BOM Items    : {len(plan.bom)}")
    print(f"  Operations   : {len(plan.operations)}")
    print(f"  Insp Points  : {len(plan.inspection_plan)}")
    print(f"  DFM Risks    : {len(plan.dfm_risks)}")
    print(f"  Inconsistencies: {len(plan.drawing_inconsistencies)}")
    print(f"  Est. Cost    : ₹ {plan.cost_estimate.get('total_cost', 0):,.0f}")
    print(f"  Confidence   : {plan.analysis_confidence*100:.1f}%")
    print("─"*65)
    if plan.drawing_inconsistencies:
        print("\n  ⚠ DRAWING INCONSISTENCIES:")
        for note in plan.drawing_inconsistencies:
            print(f"    • {note}")
    if plan.dfm_risks:
        critical = [r for r in plan.dfm_risks if r.severity in ("Critical", "High")]
        if critical:
            print("\n  🔴 CRITICAL / HIGH DFM RISKS:")
            for r in critical:
                print(f"    [{r.severity}] {r.description}")
    print(f"\n✅ OUTPUTS SAVED TO: {out_dir.resolve()}")
    print(f"   📊 Excel  : {excel_path.name}")
    print(f"   📋 JSON   : {json_path.name}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python manufacturing_ai.py <drawing.pdf>")
        print("Example: python manufacturing_ai.py shaft_drawing.pdf")
        sys.exit(1)
    run(sys.argv[1])

# Assessment Workflow Overview

This document explains how the Intelligent Vehicle External Damage Assessment and Cost Estimator works, from image upload to final repair estimate.

## System Overview

The system consists of four main components working in sequence:

1. **Image Upload & Preprocessing** - User uploads damage photos with vehicle details
2. **Damage Detection** - YOLO11s-seg instance segmentation model identifies damage regions
3. **Severity Assessment & Cost Estimation** - Core logic grades damage and calculates repair costs
4. **Report Generation** - Itemised estimate with GST and PDF report

## Step-by-Step Workflow

### 1. Image Upload & Preprocessing
- User uploads one or more photographs of vehicle damage
- System captures vehicle details (make, model, year, registration, owner info)
- User specifies which panel/component is shown in each image (editable pre-selection based on angle)
- Images are validated for format (JPEG), size (<10 MB), and minimum dimensions (>320px)

### 2. Damage Detection
The core detection pipeline uses a YOLO11s-seg (segmentation) model trained on the CarDD dataset:

```
Input Image → YOLO11s-seg Model → Detection Results
                              ↓
                     Polygon Masks + Class Labels + Confidence Scores
```

**Key Technical Details:**
- **Model**: YOLO11s-seg (10,069,525 parameters, 32.9 GFLOPs)
- **Training**: 100 epochs on 3,933 training images (2.321 hours on Colab T4)
- **Input Size**: 640×640 pixels
- **Output**: For each detection:
  - Class label (scratch, dent, crack, lamp_broken, dislocated_part, glass_shatter, tire_flat)
  - Confidence score (0-1)
  - Polygon mask defining precise damage boundaries
  - Bounding box coordinates

### 3. Severity Assessment & Cost Estimation

#### 3.1 Area Calculation
For each detection, the system calculates the damage area in pixels:
- Uses rasterization of the segmentation polygon for exact pixel count
- Critical fix: `return float(canvas.histogram()[1])` (was `[-1]`) to get foreground pixel count

#### 3.2 Reference Area & Normalization
Damage severity is assessed relative to a reference area:
- **Default**: Full image area (width × height)
- **Optional**: Vehicle mask area from COCO-pretrained model (for scale invariance)
- Area fraction = damage area / reference area (clamped to 0-1)

#### 3.3 Class-Specific Metrics
Recognizing that area alone is insufficient for certain damage types, the system uses different geometric features:

| Damage Type      | Primary Metric           | Rationale                                      |
|------------------|--------------------------|------------------------------------------------|
| scratch, crack   | Approximate length       | Better correlates with repaint effort          |
| dent, dislocated_part | Area                | Often reasonable for these damage types        |
| others           | Area fraction            | Default approach                               |

**Length Approximation**: For linear damages, polygon perimeter is calculated by summing Euclidean distances between consecutive vertices.

#### 3.4 Severity Banding
Using the selected metric (area fraction or length-based), damages are banded into:
- **Minor**: Below class-specific minor_max threshold
- **Moderate**: Between minor_max and moderate_max  
- **Severe**: Above moderate_max threshold

Thresholds are defined in `core/damage_config.py` and incorporate domain knowledge (e.g., shattered glass is never minor).

#### 3.5 Cost Calculation
Repair cost combines parts, labour, and paint materials:

```
Cost = [Part Cost × Segment Multiplier × Replace Fraction] 
     + [Labour Rate × Labour Hours] 
     + [Paint Cost × Segment Multiplier (if paintable)]
     + [GST at 18%]
```

Where:
- **Part Cost**: Baseline for hatchback segment, scaled by vehicle segment
- **Segment Multiplier**: Adjusts for vehicle type (hatchback=1.0, sedan=1.25, SUV=1.60, luxury=3.0)
- **Replace Fraction**: 0.0 (repair) to 1.0 (replace part)
- **Labour Rate**: INR/hour based on vehicle segment
- **Labour Hours**: Hours required for repair action
- **Paint Cost**: Material cost (only for paintable panels)

#### 3.6 Multi-View Merge
When multiple images show the same panel:
- System takes the **maximum** measured area (not sum)
- Conservative approach: avoids inventing damage that isn't there
- Cannot re-identify damage across views, so treats multiple photos of same damage as one repair

### 4. Report Generation
Final output includes:
- Itemised table showing each detected damage
- Per-damage: class, severity, panel, cost breakdown (parts, labour, paint)
- Totals: subtotal, GST (18%), final amount
- Overall vehicle severity (worst of all detections)
- PDF generation with annotated images

## Key Design Decisions

### 1. Area-Based Severity with Class-Specific Adjustments
**Why**: The training data has damage *types* but no severity labels, so severity cannot be learned directly. Area is an objective, measurable proxy for repair work.

**Limitation**: For 84% of damage types (scratch, crack, dent, dislocated_part), polygon area is unreliable even when detection is correct, due to annotator disagreement on boundaries.

**Mitigation**: Use length-based metrics for linear damages where area poorly correlates with repair effort.

### 2. Maximum Rather Than Sum for Multi-View Merging
**Why**: System cannot determine if damage in multiple photos is the same instance or separate damage. Taking maximum avoids overcharging for the same damage.

**Trade-off**: May undercharge for genuinely separate damage, but this is preferred over overcharging.

### 3. User-Specified Panel Rather Than Predicted
**Why**: Damage *types* don't indicate which panel is damaged. Part prices depend entirely on the damaged component.

**Approach**: Ask user to specify panel (pre-selected by image angle but editable) for honest and accurate pricing.

## Performance Metrics

The model's detection performance on the held-out test split (603 images, 1,299 instances):

### Overall Performance
| Metric       | Box    | **Mask** |
|--------------|--------|----------|
| mAP50        | 0.575  | **0.554**|
| mAP50-95     | 0.427  | **0.401**|

*Note: Mask mAP is reported because severity is computed from polygon area, making mask quality the relevant metric.*

### Per-Class Performance (Mask)
Ordered worst to best by mAP50:

| Class             | mAP50 | mAP50-95 | Boundary* | Share | Test Instances |
|-------------------|-------|----------|-----------|-------|----------------|
| crack             | 0.192 | 0.066    | 0.34      | 10.4% | 135            |
| dislocated_part   | 0.288 | 0.117    | 0.41      | 12.4% | 161            |
| scratch           | 0.407 | 0.185    | 0.45      | 35.3% | 458            |
| dent              | 0.501 | 0.253    | 0.50      | 26.1% | 339            |
| lamp_broken       | 0.739 | 0.586    | 0.79      | 6.5%  | 85             |
| glass_shatter     | 0.788 | 0.677    | 0.86      | 6.9%  | 89             |
| tire_flat         | ≈0.96 | ≈0.92    | 0.96      | 2.5%  | 32             |

*Boundary = mAP50-95 ÷ mAP50 - measures how well detection survives stricter overlap threshold*

## Accuracy Limitations & Interpretation

### Known Limitations
1. **Area Unreliability**: 84% of damage instances have unreliable polygon area even with correct detection
2. **Training Volume Paradox**: Accuracy negatively correlates with training data (Spearman ρ = −0.64)
3. **Scale Dependence**: Area fraction changes with camera distance unless vehicle mask is used as reference

### Defensible Position for Viva
The system demonstrates a **working end-to-end pipeline** where:
- Detection stage is sound (measurable mAP metrics)
- Area-to-cost stage inherits a known, measured boundary error
- Claims are backed by measurements, not assertions
- Limitations are clearly stated and quantified

### What Does Not Predict Accuracy
- **Training data volume**: More data for weak classes (scratch) doesn't help due to inherent ambiguity
- **Model complexity**: Architecture improvements don't resolve annotator disagreement on boundaries

### What Does Predict Accuracy
- **Boundary definability**: Classes with clear edges (tire_flat, lamp_broken, glass_shatter) perform better
- **Repair action consistency**: Discrete objects with agreed-upon extent yield reliable polygons

## Example Assessment Flow

**Scenario**: User uploads `scratch.jpg` showing a deep scratch on rear fender

1. **Upload**: Image + vehicle details (Maruti Swift, 2019, hatchback)
2. **Detection**: Model finds one scratch polygon (85,059 pixels area)
3. **Area Fraction**: 85,059 / (640×480) = 0.277 (27.7% of image)
4. **Severity Assessment**: 
   - For scratch: use length metric (not area)
   - Length ≈ 1334 pixels, characteristic size ≈ 529 pixels (bbox diagonal)
   - Length metric = 1334/529 = 2.52 (252% of reference)
   - Exceeds severe threshold → graded as severe
5. **Cost Estimation**:
   - Repair action: "Full panel re-paint and blend"
   - Labour: 3.5 hours × ₹450/h = ₹1,575
   - Paint: ₹2,500 (materials)
   - Parts: ₹0 (repair in place)
   - Subtotal: ₹4,075
   - GST (18%): ₹733.50
   - **Total: ₹4,808.50**
6. **Output**: Itemised estimate with PDF report

## Verification & Testing

The system includes comprehensive automated tests:
- **Unit Tests**: Core logic (severity, cost, geometry) in `tests/`
- **Regression Baseline**: 155-case cost history to detect unintended changes
- **Multi-View Properties**: Ensures consistent behavior across image angles
- **AST Checkers**: Verify all template/admin/API model accesses are valid

Run all tests with: `python tests/run_all.py` (~1 second, no GPU/db/network)

## Future Improvements

1. **Scale Invariance**: Integrate COCO vehicle masks for reference area
2. **Temporal Consistency**: Video-based assessment for better boundary estimation
3. **Uncertainty Quantification**: Provide confidence intervals on estimates
4. **Explainability**: Show which pixels contributed to severity decision
5. **Active Learning**: Flag uncertain predictions for human review

## References

See `docs/RESULTS.md` for full evaluation metrics and methodology.
See `core/damage_config.py` for severity thresholds and cost rates.
See `core/severity.py` for severity assessment logic.
See `core/cost.py` for pricing model and multi-view merge.
# Protocol Metrics Documentation

## Overview

This documentation explains the comprehensive metrics system for analyzing pipetting video data, specifically designed for reproducibility experiments comparing different operator types (freshly trained students, experienced lab workers, and automated liquid handlers) across a standardized liquid handling protocol.

## Table of Contents

1. [Available Metrics](#available-metrics)
2. [Operator Comparison Guidelines](#operator-comparison-guidelines)
3. [Video Comparison Guidelines](#video-comparison-guidelines)
4. [Interpretation Guidelines](#interpretation-guidelines)
5. [Using the Plotting System](#using-the-plotting-system)
6. [Data Structure Reference](#data-structure-reference)

---

## Available Metrics

### 1. Protocol Execution Metrics

#### **Protocol Completion**
- **Description**: Percentage of expected protocol steps completed
- **Range**: 0-100% (can exceed 100% if extra cycles detected)
- **Target**: 100% for complete execution
- **Example**: `protocol_completion_percentage: 443.8%` indicates multiple protocol repetitions

#### **Cycle Detection**
- **estimated_cycles**: Number of pipetting cycles detected
- **total_aspiration_events**: Count of liquid aspiration actions
- **total_dispensing_events**: Count of liquid dispensing actions
- **total_tip_changes**: Count of pipette tip replacement events

#### **Timing Analysis**
- **total_time_seconds**: Complete protocol execution time
- **average_cycle_time**: Mean time per pipetting cycle
- **cycle_consistency_score**: Consistency of timing across cycles (0-1, higher = more consistent)
- **time_efficiency_score**: Efficiency relative to expected completion time

### 2. Hand Detection & Movement Metrics

#### **Detection Performance**
- **hand_detection_rate**: Percentage of frames with hands detected (target: >90%)
- **avg_confidence_dominant**: Average confidence score for dominant hand detection
- **frames_with_hands**: Total frames where hands were successfully detected

#### **Movement Analysis**
- **dominant_hand_velocity**: Speed of dominant hand movement (pixels/frame)
- **movement_smoothness**: Inverse of velocity standard deviation (higher = smoother)
- **spatial_variability**: Consistency of hand positions across cycles
- **velocity_variability**: Consistency of movement speeds

### 3. Reproducibility Metrics

#### **Overall Reproducibility Score** (0-1 scale)
- **Calculation**: Combined score from timing, spatial, and velocity consistency
- **Interpretation**:
  - `>0.9`: Automated/expert level consistency
  - `0.7-0.9`: Experienced operator level
  - `0.5-0.7`: Trained student level
  - `<0.5`: Novice/learning level

#### **Inter-Cycle Variability**
- **timing_variability**: Standard deviation of cycle times (seconds)
- **spatial_variability**: Variation in hand position patterns
- **velocity_variability**: Variation in movement speed patterns

### 4. Technique Adherence Scores (0-1 scale)

#### **Liquid-Specific Techniques**
- **reverse_pipetting_glycerol**: Adherence to reverse pipetting for viscous liquids
- **pre_wetting_ethanol**: Pre-wetting technique for volatile liquids
- **tip_change_technique**: Proper tip changing between liquids
- **cross_contamination_prevention**: Overall contamination prevention

### 5. Accuracy Predictions

#### **Volume-Specific Predictions**
For each volume (50μL, 100μL, 150μL, 200μL):
- **predicted_cv_percent**: Predicted coefficient of variation (%)
- **target_cv_percent**: Protocol target CV (decreases with volume)
- **meets_target**: Boolean indicating if prediction meets accuracy target
- **accuracy_score**: Normalized accuracy score (0-1)

**Accuracy Targets by Volume:**
- 50μL: <5% CV
- 100μL: <3% CV  
- 150μL: <2.5% CV
- 200μL: <2% CV

### 6. Operator Classification

#### **Automated Assessment**
- **operator_type_classification**: Automated categorization based on performance
- **estimated_experience_level**: Human-readable experience assessment
- **overall_skill_score**: Combined skill assessment (0-1)

**Classification Logic:**
- **Automated Handler**: Reproducibility >0.9, Timing consistency >0.9
- **Experienced Worker**: Technique adherence >0.8, Timing consistency >0.7
- **Trained Student**: Technique adherence >0.6, Reproducibility >0.5
- **Novice**: Below trained student thresholds

---

## Operator Comparison Guidelines

### Step 1: Analyze Individual Operators

```bash
# Run analysis for each operator type
python reproducibility_experiment.py video1.mp4 --operator-type freshly_trained_student --operator-id student01
python reproducibility_experiment.py video2.mp4 --operator-type experienced_lab_worker --operator-id worker01
python reproducibility_experiment.py video3.mp4 --operator-type automated_liquid_handler --operator-id robot01
```

### Step 2: Generate Comparison Analysis

```bash
# Generate comparison plots and reports
python reproducibility_experiment.py dummy.mp4 --operator-type freshly_trained_student --compare
```

### Step 3: Key Comparison Metrics

#### **Primary Reproducibility Comparison**
- Compare `overall_reproducibility_score` across operator types
- Expected hierarchy: Automated > Experienced > Student

#### **Accuracy Performance**
- Compare `predicted_cv_percent` by volume
- Analyze `meets_target` rates by operator type
- Focus on volume-dependent performance differences

#### **Technique Adherence**
- Compare technique scores for liquid-specific handling
- Identify systematic differences in contamination prevention
- Assess tip-changing consistency

#### **Temporal Consistency** 
- Compare `cycle_consistency_score` across operators
- Analyze timing variability patterns
- Identify learning curves or fatigue effects

### Step 4: Statistical Analysis

#### **Box Plot Interpretation**
- Median values indicate central tendency by operator type
- Interquartile ranges show consistency within operator groups
- Outliers indicate individual performance variations

#### **Scatter Plot Analysis**
- Efficiency vs. Consistency plots reveal operator strategies
- Volume-dependent accuracy shows skill scaling
- Technique adherence patterns indicate training effectiveness

---

## Video Comparison Guidelines

### Comparing Multiple Videos from Same Operator

#### **Intra-Operator Variability**
```python
# Example analysis for operator consistency
operator_videos = [
    "operator1_day1.mp4",
    "operator1_day2.mp4", 
    "operator1_day3.mp4"
]

# Key metrics to compare:
# - reproducibility_score variation
# - cycle_time consistency
# - technique_adherence stability
```

#### **Learning Curve Analysis**
- Track `overall_skill_score` across sessions
- Monitor `cycle_consistency_score` improvements
- Analyze `technique_adherence` score changes

### Comparing Different Protocols

#### **Protocol Complexity Assessment**
- Compare `protocol_completion_percentage` across protocols
- Analyze `estimated_cycles` vs expected cycles
- Assess `technique_adherence` requirements

#### **Environmental Factors**
- Compare `hand_detection_rate` across recording conditions
- Analyze `avg_confidence_dominant` for lighting/glove effects
- Consider camera angle and setup variations

---

## Interpretation Guidelines

### Performance Categories

#### **Excellent Performance (Score: 0.8-1.0)**
- **Reproducibility**: >0.8
- **Accuracy**: Meets all volume targets
- **Technique**: >0.8 for all liquid-specific techniques
- **Timing**: Consistent cycle times (CV <20%)

#### **Good Performance (Score: 0.6-0.8)**
- **Reproducibility**: 0.6-0.8
- **Accuracy**: Meets 75% of volume targets
- **Technique**: >0.6 for critical techniques
- **Timing**: Moderate consistency (CV 20-40%)

#### **Needs Improvement (Score: <0.6)**
- **Reproducibility**: <0.6
- **Accuracy**: Meets <50% of volume targets  
- **Technique**: <0.6 for critical techniques
- **Timing**: High variability (CV >40%)

### Red Flags for Data Quality

#### **Technical Issues**
- `hand_detection_rate < 85%`: Poor video quality or glove detection issues
- `avg_confidence_dominant < 0.5`: Lighting or hand visibility problems
- `processed_frames` very low: Video processing errors

#### **Protocol Issues**
- `protocol_completion_percentage < 50%`: Incomplete or interrupted protocol
- `estimated_cycles = 0`: No clear pipetting cycles detected
- `total_aspiration_events = 0`: No liquid handling detected

### Benchmark Comparisons

#### **Expert Laboratory Standards**
- Reproducibility score: >0.85
- Cycle time CV: <15%
- Accuracy predictions: Meet all volume targets
- Technique adherence: >0.9 for all techniques

#### **Trained Student Expectations**
- Reproducibility score: 0.6-0.8
- Cycle time CV: 15-30%
- Accuracy predictions: Meet 60-80% of targets
- Technique adherence: >0.7 for critical techniques

#### **Automated System Baselines**
- Reproducibility score: >0.95
- Cycle time CV: <5%
- Accuracy predictions: Meet all targets with margin
- Technique adherence: Perfect scores where applicable

---

## Using the Plotting System

### Generating Individual Analysis Plots

```python
from protocol_metrics_plotter_fixed import ProtocolMetricsPlotter

# Create plotter instance
plotter = ProtocolMetricsPlotter(
    data_path="analysis_data.json",
    output_dir="plots/"
)

# Generate all standard plots
plots = plotter.generate_all_plots()
```

### Available Plot Types

#### **1. Protocol Overview Plot**
- **File**: `protocol_overview.png`
- **Content**: Key performance indicators, completion status, timing summary
- **Use**: Quick assessment of overall performance

#### **2. Volume Analysis Plot**
- **File**: `volume_analysis.png` 
- **Content**: Accuracy predictions by volume, target comparisons, technique scores
- **Use**: Detailed accuracy and technique assessment

#### **3. Temporal Analysis Plot**
- **File**: `temporal_analysis.png`
- **Content**: Hand detection rates, velocity patterns, movement smoothness over time
- **Use**: Understanding performance changes during protocol execution

### Comparison Plots (Multi-Operator)

#### **Reproducibility Comparison**
- Box plots of reproducibility scores by operator type
- Volume-specific accuracy scatter plots
- Timing consistency vs. efficiency analysis
- Technique adherence radar charts

---

## Data Structure Reference

### JSON Structure Overview

```json
{
  "analysis_timestamp": "2025-06-17T01:28:25",
  "protocol_config": { ... },
  "metrics": {
    "video_info": { ... },
    "hand_detection": { ... },
    "protocol_analysis": {
      "protocol_events": { ... },
      "timing_metrics": { ... },
      "skill_assessment": { ... },
      "reproducibility_analysis": {
        "reproducibility_metrics": { ... },
        "accuracy_predictions": { ... }
      }
    }
  }
}
```

### Key Data Paths

#### **Performance Metrics**
- `metrics.protocol_analysis.reproducibility_analysis.overall_reproducibility_score`
- `metrics.protocol_analysis.timing_metrics.cycle_consistency_score`
- `metrics.protocol_analysis.skill_assessment.overall_skill_score`

#### **Accuracy Data**
- `metrics.protocol_analysis.reproducibility_analysis.accuracy_predictions[volume].predicted_cv_percent`
- `metrics.protocol_analysis.reproducibility_analysis.accuracy_predictions[volume].meets_target`

#### **Technique Scores**
- `metrics.protocol_analysis.reproducibility_analysis.reproducibility_metrics.technique_adherence.reverse_pipetting_glycerol`
- `metrics.protocol_analysis.reproducibility_analysis.reproducibility_metrics.technique_adherence.pre_wetting_ethanol`

### CSV Frame Data

Frame-by-frame data is saved in CSV format with columns:
- `frame_number`, `time_seconds`: Temporal reference
- `hands_detected`, `dom_hand_confidence`: Detection metrics
- `dom_hand_x`, `dom_hand_y`: Hand position coordinates  
- `dominant_hand_velocity`: Movement speed
- Additional protocol-specific columns

---

## Best Practices

### Data Collection
1. **Consistent Setup**: Use identical camera angles, lighting, and background
2. **Glove Optimization**: Use hue_offset=135 for optimal glove detection
3. **Complete Protocols**: Ensure full protocol execution for meaningful comparisons
4. **Multiple Replicates**: Collect 3-5 videos per operator for statistical validity

### Analysis Workflow
1. **Individual Analysis**: Process each video separately first
2. **Quality Check**: Verify hand detection rates >85% before comparison
3. **Comparison Analysis**: Use the comparison tools only after individual analyses
4. **Statistical Testing**: Apply appropriate statistical tests for group comparisons

### Reporting
1. **Context**: Always include operator type, experience level, and protocol details
2. **Limitations**: Note any technical issues or incomplete data
3. **Confidence**: Report detection confidence and data quality metrics
4. **Recommendations**: Provide actionable insights based on metric patterns

---

## Troubleshooting Common Issues

### Low Hand Detection Rates
- **Problem**: `hand_detection_rate < 85%`
- **Solutions**: 
  - Adjust `hue_offset` parameter for glove color
  - Improve lighting conditions
  - Check for hand obstructions
  - Consider camera angle adjustments

### Inconsistent Cycle Detection
- **Problem**: `estimated_cycles` much higher/lower than expected
- **Solutions**:
  - Verify protocol execution completeness
  - Check for false positive/negative event detection
  - Adjust event detection thresholds
  - Review frame sampling rate

### Poor Reproducibility Scores
- **Problem**: `overall_reproducibility_score < 0.5` for experienced operators
- **Solutions**:
  - Increase frame sampling resolution (lower skip_frames)
  - Check for protocol variations or interruptions
  - Verify operator followed standard procedure
  - Consider individual operator variation

This documentation provides a comprehensive guide for understanding and using the protocol metrics system for reproducibility experiments. For specific questions or advanced use cases, refer to the code comments in the respective analysis modules.

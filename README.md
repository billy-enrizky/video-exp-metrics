# Video-Based Pipetting Technique Analysis Metrics

This repository contains a comprehensive video analysis system for evaluating pipetting techniques in liquid handling experiments. The system analyzes operator performance through computer vision and generates quantitative metrics for reproducibility assessment.

## Protocol Overview

**Objective**: To evaluate and compare the accuracy, reproducibility, and time efficiency of liquid dispensing between automated liquid handlers and manual pipetting across different operator skill levels.

**Protocol**: 4 volumes (50, 100, 150, 200 μL) × 4 liquids = 16 total dispensing events
- **DYE ddH₂O**: Standard water with dye (3 dispensing events per volume)
- **DYE-FREE ddH₂O**: Pure water (standard technique)
- **GLYCEROL**: Viscous liquid requiring reverse pipetting
- **ETHANOL**: Volatile liquid requiring pre-wetting technique

## Comprehensive Metrics Documentation

### 1. Protocol Event Detection Metrics

#### 1.1 Aspiration Event Detection
**Formula**: `total_aspiration_events = count(upward_movement_magnitude > 15 AND confidence > 0.8)`

**English Explanation**: Counts upward hand movements above threshold that indicate liquid aspiration into the pipette tip.

**Reasoning**: 
- **Threshold 15 pixels**: Chosen based on typical pipette movement amplitude in video resolution
- **Confidence 0.8**: High confidence threshold ensures we only count genuine hand movements, reducing false positives from detection noise

#### 1.2 Dispensing Event Detection
**Formula**: `total_dispensing_events = count(downward_movement_magnitude > 15 AND confidence > 0.8)`

**English Explanation**: Counts downward hand movements that indicate liquid dispensing from the pipette.

**Reasoning**: Same movement and confidence thresholds as aspiration, but for downward movements indicating dispensing actions.

#### 1.3 Tip Change Event Detection
**Formula**: `tip_change_events = count(velocity < mean_velocity * 0.05 for extended_period)`

**English Explanation**: Detects periods of minimal hand movement indicating tip disposal and replacement.

**Reasoning**: 
- **5% of mean velocity**: Very low activity threshold indicates hands are stationary during tip changes
- Prevents counting brief pauses as tip changes while capturing actual equipment manipulation

### 2. Timing Consistency Metrics

#### 2.1 Cycle Consistency Score
**Formula**: `cycle_consistency_score = 1 - (std(cycle_times) / mean(cycle_times))`

**English Explanation**: Measures how consistent the operator is in timing between repeated pipetting cycles. Higher scores indicate better reproducibility.

**Reasoning**: 
- **Coefficient of Variation inversion**: CV measures relative variability; subtracting from 1 makes higher values better
- **Range 0-1**: Provides intuitive scoring where 1.0 = perfect consistency, 0.0 = highly variable

#### 2.2 Time Efficiency Score
**Formula**: `time_efficiency_score = expected_completion_time / actual_total_time`

**English Explanation**: Compares actual protocol completion time against expected duration (20 minutes = 1200 seconds).

**Reasoning**: 
- **Expected time 1200s (20 min)**: Based on protocol design and expert performance benchmarks
- **Ratio format**: Values >1.0 indicate faster than expected, <1.0 indicate slower than expected

### 3. Technique-Specific Adherence Metrics

#### 3.1 Reverse Pipetting Adherence (Glycerol)
**Formula**: `reverse_pipetting_score = min(1.0, detected_extra_steps / expected_extra_steps)`
- Where `expected_extra_steps = 2` and `expected_direction_changes = 12`

**English Explanation**: Measures adherence to reverse pipetting technique required for viscous liquids like glycerol.

**Reasoning**: 
- **Extra steps = 2**: Reverse pipetting requires additional aspiration beyond first stop, creating 2 extra movement phases
- **Direction changes = 12**: More movement reversals due to the complex aspiration pattern vs. 8 for standard technique
- **Technique necessity**: Glycerol's high viscosity requires this specialized technique for accuracy

#### 3.2 Pre-wetting Adherence (Ethanol)
**Formula**: `pre_wetting_score = 1.0 if direction_changes >= 12 else 0.7 if >= 8 else 0.3`

**English Explanation**: Assesses whether the operator performed pre-wetting cycles before ethanol aspiration to prevent tip dripping.

**Reasoning**: 
- **12 direction changes**: Pre-wetting requires initial aspiration-dispense cycles, doubling normal movement patterns
- **Graduated scoring**: Accounts for partial adherence (some pre-wetting) vs. complete adherence
- **Ethanol volatility**: High vapor pressure causes tip dripping without pre-wetting, affecting accuracy

### 4. Spatial Movement Metrics

#### 4.1 Spatial Variability
**Formula**: `spatial_variability = std(distances_from_center)`
- Where `distances_from_center = sqrt((x - center_x)² + (y - center_y)²)`

**English Explanation**: Measures how much hand position varies from the average working position during protocol execution.

**Reasoning**: 
- **Euclidean distance**: Standard geometric measure of position deviation
- **Center-based**: Uses mean position as reference to account for individual working preferences
- **Lower values**: Indicate more consistent, controlled movements

#### 4.2 Hand Steadiness Score
**Formula**: `steadiness_score = 1.0 - (velocity_std / (velocity_mean + 0.001))`

**English Explanation**: Quantifies hand stability during pipetting operations, with higher scores indicating steadier hands.

**Reasoning**: 
- **CV-based measure**: Relative variability accounts for different movement speeds
- **Epsilon term (+0.001)**: Prevents division by zero in perfectly still moments
- **Inversion (1.0 -)**: Makes higher values represent better steadiness

### 5. Volume-Specific Consistency Metrics

#### 5.1 Duration Consistency CV
**Formula**: `duration_cv = (std(cycle_durations) / mean(cycle_durations)) * 100`

**English Explanation**: Coefficient of variation for time taken to complete each volume dispensing cycle.

**Reasoning**: 
- **Percentage format**: Standard CV presentation for easy interpretation
- **Per-volume analysis**: Different volumes require different timing, so consistency is measured within each volume group
- **Reproducibility indicator**: Lower CV indicates more reproducible technique

#### 5.2 Velocity Consistency CV
**Formula**: `velocity_cv = (std(avg_velocities) / mean(avg_velocities)) * 100`

**English Explanation**: Measures consistency in movement speed across cycles for each volume.

**Reasoning**: 
- **Movement speed**: Different volumes may require different speeds, but consistency within volume is crucial
- **Technical skill indicator**: Experienced operators maintain consistent velocities for each volume

### 6. Accuracy Prediction Metrics

#### 6.1 Predicted CV Percentage
**Formula**: `predicted_cv = (velocity_cv * 0.6 + duration_cv * 0.4) / 5.0`

**English Explanation**: Predicts the coefficient of variation in actual dispensed volumes based on movement consistency.

**Reasoning**: 
- **Velocity weight 0.6**: Movement consistency is primary predictor of pipetting accuracy
- **Duration weight 0.4**: Timing consistency contributes but is secondary to movement control
- **Scale factor 5.0**: Empirical scaling to match typical pipetting CV ranges (1-10%)

#### 6.2 Accuracy Target Assessment
**Formula**: `meets_target = predicted_cv <= target_cv`
- Target CVs: 50μL < 5%, 100μL < 3%, 150μL < 2.5%, 200μL < 2%

**English Explanation**: Determines if predicted performance meets volume-specific accuracy standards.

**Reasoning**: 
- **Volume-dependent targets**: Larger volumes typically achieve better relative precision
- **Industry standards**: Based on published pipetting accuracy requirements for laboratory work
- **50μL target 5%**: Smallest volume has highest relative error tolerance
- **200μL target 2%**: Largest volume should achieve highest relative precision

### 7. Operator Classification Metrics

#### 7.1 Overall Reproducibility Score
**Formula**: `overall_score = mean([timing_score, spatial_score, velocity_score])`
Where:
- `timing_score = 1.0 - min(timing_variability / 10.0, 1.0)`
- `spatial_score = 1.0 - min(spatial_variability / 100.0, 1.0)`
- `velocity_score = 1.0 - min(velocity_variability / 50.0, 1.0)`

**English Explanation**: Composite score combining timing, spatial, and velocity consistency into overall reproducibility assessment.

**Reasoning**: 
- **Normalization factors**: 10.0s timing, 100px spatial, 50px/frame velocity represent maximum acceptable variability
- **Equal weighting**: All three components equally important for overall technique assessment
- **0-1 scale**: Provides intuitive scoring system

#### 7.2 Operator Type Classification
**Formula**: 
```
if reproducibility > 0.9 AND timing_consistency > 0.9:
    return "automated_liquid_handler"
elif technique_adherence > 0.8 AND timing_consistency > 0.7:
    return "experienced_lab_worker"  
elif technique_adherence > 0.6 AND reproducibility > 0.5:
    return "freshly_trained_student"
else:
    return "novice_learning"
```

**English Explanation**: Classifies operator skill level based on consistency and technique adherence thresholds.

**Reasoning**: 
- **Automated handler thresholds (0.9, 0.9)**: Machines should show near-perfect consistency
- **Experienced worker thresholds (0.8, 0.7)**: High technique knowledge with good consistency
- **Trained student thresholds (0.6, 0.5)**: Follows protocols but with moderate consistency
- **Progressive thresholds**: Reflect realistic performance expectations for each skill level

### 8. Contamination Prevention Metrics

#### 8.1 Cross-Contamination Prevention Score
**Formula**: `contamination_score = min(1.0, low_activity_periods / (total_periods * 0.1))`

**English Explanation**: Measures frequency of low-activity periods indicating proper contamination prevention procedures.

**Reasoning**: 
- **10% target**: Expect ~10% of time spent on contamination prevention activities
- **Low activity detection**: Periods of minimal movement indicate tip changes, cleaning, or waiting
- **Protocol requirement**: Essential for multi-liquid protocols to prevent cross-contamination

#### 8.2 Tip Change Technique Score
**Formula**: `tip_change_score = min(1.0, detected_tip_changes / expected_tip_changes)`
- Where `expected_tip_changes = 4` (one per liquid type)

**English Explanation**: Assesses whether operator changed tips appropriately between different liquids.

**Reasoning**: 
- **4 expected changes**: Protocol requires fresh tip for each liquid (DYE ddH₂O, DYE-FREE ddH₂O, GLYCEROL, ETHANOL)
- **Detection method**: Extended periods of minimal movement followed by resumed activity
- **Contamination prevention**: Critical for maintaining liquid purity and experimental validity

### 9. Hand Detection Technical Metrics

#### 9.1 Hand Detection Rate
**Formula**: `detection_rate = frames_with_hands / total_processed_frames`

**English Explanation**: Percentage of video frames where hands were successfully detected by computer vision system.

**Reasoning**: 
- **Quality indicator**: Higher detection rates indicate better video quality and algorithm performance
- **Confidence threshold**: Uses detection confidence > 0.2 to balance sensitivity vs. false positives
- **Glove compatibility**: Adjusted thresholds for gloved hand detection scenarios

#### 9.2 Detection Confidence Score
**Formula**: `avg_confidence = mean(detection_confidence_scores)`

**English Explanation**: Average confidence level of hand detection across all frames.

**Reasoning**: 
- **Algorithm reliability**: Higher confidence indicates more reliable tracking
- **Range 0-1**: MediaPipe confidence scores provide quality assessment
- **Threshold selection**: 0.2 minimum chosen to include gloved hands while excluding noise

## Analysis Pipeline Summary

The complete analysis pipeline processes video data through:

1. **Frame-by-frame hand detection** with glove-optimized parameters
2. **Movement pattern analysis** to identify protocol events
3. **Timing analysis** to measure cycle consistency
4. **Technique assessment** for specialized liquid handling
5. **Statistical analysis** to predict accuracy and classify operators
6. **Comprehensive reporting** with visual analytics

Each metric is designed to capture specific aspects of pipetting technique that correlate with actual laboratory performance, enabling objective assessment of operator skill and protocol adherence.
# Video-Based Pipetting Technique Analysis Metrics

This repository contains a comprehensive video analysis system for evaluating pipetting techniques in liquid handling experiments. The system analyzes operator performance through computer vision and generates quantitative metrics for reproducibility assessment.

## Protocol Overview

**Objective**: To evaluate and compare the accuracy, reproducibility, and time efficiency of liquid dispensing between automated liquid handlers and manual pipetting across different operator skill levels.

**Complete Experimental Context**: This analysis system was developed for a comprehensive liquid handling study comparing manual pipetting across various people.

### Detailed Protocol Description

**Experiment 1: Basic Liquid Dispensing Protocol**
- **Protocol**: 4 volumes (50, 100, 150, 200 μL) × 4 liquids = 16 total dispensing events
- **Equipment**: Single-channel P200 pipette, P200 tips, 96-well plates, precision timer
- **Execution**: Start timer → complete all dispensing → stop timer (typically 15-25 minutes)

**The Four Liquid Types and Their Challenges**:

1. **DYE ddH₂O (Standard water with dye)**:
   - **Purpose**: Baseline measurement with visible liquid for accuracy verification
   - **Technique**: Standard pipetting (aspirate to first stop, dispense to first stop)
   - **Volumes**: 50, 100, 150, 200 μL dispensed 3 times each = 12 dispensing events
   - **Challenge Level**: Easy - standard aqueous solution

2. **DYE-FREE ddH₂O (Pure water)**:
   - **Purpose**: Control for dye interference, tests technique without visual aid
   - **Technique**: Standard pipetting
   - **Volumes**: 50, 100, 150, 200 μL dispensed once each = 4 dispensing events
   - **Challenge Level**: Moderate - harder to see dispensed volume

3. **GLYCEROL (Viscous liquid)**:
   - **Purpose**: Tests handling of high-viscosity liquids (glycerol ~1000× more viscous than water)
   - **Technique**: **Reverse pipetting required**
   - **Volumes**: 50, 100, 150, 200 μL dispensed once each = 4 dispensing events
   - **Challenge Level**: High - requires specialized technique

4. **ETHANOL (Volatile liquid)**:
   - **Purpose**: Tests handling of volatile, low-surface-tension liquids
   - **Technique**: **Pre-wetting required**
   - **Volumes**: 50, 100, 150, 200 μL dispensed once each = 4 dispensing events
   - **Challenge Level**: High - prone to dripping and evaporation

### Specialized Pipetting Techniques Explained

#### Reverse Pipetting (Required for Glycerol)
**Why Needed**: Glycerol's high viscosity (1000× water) causes liquid to stick to tip walls, leading to under-dispensing with standard technique.

**Technique Steps**:
1. Press pipette button to **second stop** (blow-out position)
2. Insert tip into liquid and **aspirate** (liquid fills tip plus extra air space)
3. Move to target location
4. Press button only to **first stop** to dispense (leaves residual liquid in tip)
5. **Extra movements**: Additional aspiration and partial dispensing create complex movement patterns

**Movement Pattern Impact**: 
- **Standard technique**: ~8 direction changes (down-aspirate-up-down-dispense-up)
- **Reverse technique**: ~12 direction changes (extra aspiration steps add 4 more movements)

#### Pre-wetting (Required for Ethanol)
**Why Needed**: Ethanol's high vapor pressure causes air expansion in pipette, leading to tip dripping and volume inaccuracy.

**Technique Steps**:
1. **Initial cycles**: Aspirate and dispense ethanol 2-3 times to saturate tip interior
2. **Purpose**: Equilibrate air space with ethanol vapor to prevent expansion
3. **Then proceed**: With normal aspiration and dispensing
4. **Extra movements**: Pre-wetting cycles double the movement patterns

**Movement Pattern Impact**:
- **Standard technique**: ~8 direction changes (single aspiration-dispensing cycle)
- **Pre-wetting technique**: ~12 direction changes (extra pre-wetting cycles add 4 more movements)

### Protocol Execution Flow and Timing

**Cycle Structure**: Each "cycle" represents handling one complete liquid type through all four volumes:
- **Cycle 1**: DYE ddH₂O (50→100→150→200 μL, 3× each = 12 events)
- **Cycle 2**: DYE-FREE ddH₂O (50→100→150→200 μL, 1× each = 4 events)  
- **Cycle 3**: GLYCEROL (50→100→150→200 μL, 1× each = 4 events)
- **Cycle 4**: ETHANOL (50→100→150→200 μL, 1× each = 4 events)

**Contamination Prevention**: Fresh tip required between each liquid type (4 tip changes total)

**Expected Timing**:
- **Expert operators**: 15-18 minutes (consistent pacing, smooth technique execution)
- **Trained students**: 18-22 minutes (following protocol carefully, some hesitation)
- **Novice operators**: 22-30 minutes (frequent pauses, technique uncertainties)

## Comprehensive Metrics Documentation

### Video Analysis Coordinate System
**Image Coordinate Convention**: 
- Origin (0,0) is at top-left corner of video frame
- X-axis increases from left to right
- Y-axis increases from top to bottom (downward positive)
- Hand positions tracked as (x,y) pixel coordinates of hand center
- Movement calculations use consecutive frame differences

**Movement Magnitude Context**:
- **Typical video resolution**: 1920×1080 pixels (Full HD)
- **Hand detection area**: Usually 200-800 pixels wide depending on camera angle
- **Actual movement scale**: 15 pixels ≈ 2-5mm real-world movement (varies with camera distance)
- **Frame rate**: Typically 30 FPS, so measurements are per 1/30th second intervals

### 1. Protocol Event Detection Metrics

#### 1.1 Aspiration Event Detection
**Formula**: `total_aspiration_events = count(upward_movement_magnitude > 15 AND confidence > 0.8)`
- Where `upward_movement_magnitude = sqrt(dx² + dy²)` when `dy < 0` (upward in image coordinates)
- Units: **pixels per frame**

**English Explanation**: Counts upward hand movements above threshold that indicate liquid aspiration into the pipette tip.

**Movement Magnitude Calculation**:
- `dx = current_hand_x - previous_hand_x` (horizontal displacement)
- `dy = current_hand_y - previous_hand_y` (vertical displacement, negative = upward)
- `upward_movement_magnitude = sqrt(dx² + dy²)` only when `dy < 0`
- **Units**: Measured in pixels per frame, representing the Euclidean distance the hand center moved between consecutive video frames

**Reasoning**: 
- **Threshold 15 pixels/frame**: Based on empirical analysis of pipetting videos where typical aspiration movements show 10-30 pixel displacements in standard video resolutions (1920×1080). A 15-pixel threshold captures deliberate pipetting motions while filtering out hand tremor (typically <5 pixels) and camera shake. This corresponds to approximately 2-5mm of actual hand movement depending on camera distance and zoom level.
- **Confidence 0.8**: High confidence threshold ensures we only count genuine hand movements, reducing false positives from detection noise
- **Directional filtering**: Only upward movements (dy < 0) are counted since aspiration requires lifting the pipette from liquid surface

#### 1.2 Dispensing Event Detection
**Formula**: `total_dispensing_events = count(downward_movement_magnitude > 15 AND confidence > 0.8)`
- Where `downward_movement_magnitude = sqrt(dx² + dy²)` when `dy > 0` (downward in image coordinates)
- Units: **pixels per frame**

**English Explanation**: Counts downward hand movements that indicate liquid dispensing from the pipette.

**Movement Magnitude Calculation**:
- `dx = current_hand_x - previous_hand_x` (horizontal displacement)
- `dy = current_hand_y - previous_hand_y` (vertical displacement, positive = downward)
- `downward_movement_magnitude = sqrt(dx² + dy²)` only when `dy > 0`
- **Units**: Measured in pixels per frame, representing the Euclidean distance the hand center moved between consecutive video frames

**Reasoning**: 
- **Threshold 15 pixels/frame**: Same empirical basis as aspiration detection. Dispensing movements typically show similar magnitude displacements as the operator moves the pipette tip to the target well and depresses the plunger. The 15-pixel threshold effectively distinguishes between purposeful dispensing motions and minor positioning adjustments.
- **Confidence 0.8**: High confidence threshold ensures we only count genuine hand movements, reducing false positives from detection noise
- **Directional filtering**: Only downward movements (dy > 0) are counted since dispensing requires lowering the pipette toward the target container

#### 1.3 Tip Change Event Detection
**Formula**: `tip_change_events = count(velocity < mean_velocity * 0.05 for extended_period)`
- Where `velocity = sqrt(dx² + dy²)` (movement magnitude regardless of direction)
- Units: **pixels per frame**

**English Explanation**: Detects periods of minimal hand movement indicating tip disposal and replacement.

**Velocity Calculation**:
- `velocity = sqrt((current_x - previous_x)² + (current_y - previous_y)²)`
- **Units**: Pixels per frame, representing total hand displacement between consecutive frames
- **Extended period**: Continuous low-velocity frames lasting >2 seconds (>60 frames at 30 FPS)

**Reasoning**: 
- **5% of mean velocity**: Very low activity threshold indicates hands are stationary during tip changes. If mean velocity is 20 pixels/frame, then 5% = 1 pixel/frame represents near-stillness
- **Extended period requirement**: Prevents counting brief pauses as tip changes while capturing actual equipment manipulation phases
- **Biological basis**: Tip changes require 3-5 seconds of careful manipulation with minimal hand movement

### 2. Timing Consistency Metrics

#### 2.1 Cycle Consistency Score
**Formula**: `cycle_consistency_score = 1 - (std(cycle_times) / mean(cycle_times))`

**English Explanation**: Measures how consistent the operator is in timing between repeated pipetting cycles. Higher scores indicate better reproducibility.

**Cycle Duration Definition**:
- **What is a "cycle"**: Complete handling of one liquid type through all four volumes (50, 100, 150, 200 μL)
- **Cycle boundaries**: From first interaction with a liquid type to tip disposal before next liquid
- **Timing calculation**: `cycle_duration = time_at_tip_disposal - time_at_first_aspiration`

**Example Cycle Breakdown**:
- **DYE ddH₂O cycle**: ~8-12 minutes (12 dispensing events: 3× each volume)
- **DYE-FREE ddH₂O cycle**: ~3-5 minutes (4 dispensing events: 1× each volume)
- **GLYCEROL cycle**: ~4-7 minutes (4 dispensing events + reverse pipetting complexity)
- **ETHANOL cycle**: ~4-7 minutes (4 dispensing events + pre-wetting steps)

**Cycle Time Variability Examples**:
- **Expert**: [480s, 240s, 300s, 280s] → std = 105s → consistency = 0.69 (good)
- **Student**: [600s, 300s, 420s, 380s] → std = 127s → consistency = 0.71 (good)
- **Novice**: [720s, 480s, 600s, 540s] → std = 102s → consistency = 0.72 (surprisingly good timing despite slower pace)

**Reasoning**: 
- **Coefficient of Variation (CV) inversion**: CV measures relative variability; subtracting from 1 makes higher values better
- **Range 0-1**: Provides intuitive scoring where 1.0 = perfect consistency, 0.0 = highly variable
- **Skill assessment**: Experienced operators show more consistent cycle-to-cycle timing regardless of liquid complexity

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

**Direction Changes Detailed Breakdown**:

**Standard Pipetting Technique (8 direction changes)**:
1. **Down**: Lower pipette to liquid surface
2. **Aspirate**: Press button to first stop, release to aspirate
3. **Up**: Lift pipette from liquid  
4. **Move**: Horizontal movement to target well
5. **Down**: Lower pipette to target well
6. **Dispense**: Press button to first stop to dispense
7. **Up**: Lift pipette from well
8. **Move**: Return to rest position

**Reverse Pipetting Technique (12 direction changes)**:
1. **Down**: Lower pipette to liquid surface
2. **Pre-aspirate**: Press button to **second stop** (blow-out position)
3. **Aspirate**: Release button to aspirate extra volume
4. **Up**: Lift pipette from liquid
5. **Move**: Horizontal movement to target well  
6. **Down**: Lower pipette to target well
7. **Partial dispense**: Press button only to **first stop** (retains residual)
8. **Up**: Lift pipette from well
9. **Evaluate**: Visual check of dispensed volume
10. **Adjust**: Possible additional small dispense if needed
11. **Final up**: Complete withdrawal
12. **Move**: Return to rest position

**Reasoning**: 
- **Extra steps = 2**: Reverse pipetting requires additional aspiration beyond first stop, creating 2 extra movement phases (pre-aspirate + residual management)
- **Direction changes = 12**: More movement reversals due to the complex aspiration pattern vs. 8 for standard technique
- **Technique necessity**: Glycerol's high viscosity (η = 1.41 Pa·s vs water's 0.001 Pa·s) requires this specialized technique for accuracy
- **Detection method**: Algorithm counts velocity direction reversals to identify technique complexity

#### 3.2 Pre-wetting Adherence (Ethanol)
**Formula**: `pre_wetting_score = 1.0 if direction_changes >= 12 else 0.7 if >= 8 else 0.3`

**English Explanation**: Assesses whether the operator performed pre-wetting cycles before ethanol aspiration to prevent tip dripping.

**Direction Changes Detailed Breakdown**:

**Standard Pipetting Technique (8 direction changes)**:
1. **Down**: Lower pipette to liquid surface
2. **Aspirate**: Press button, release to aspirate
3. **Up**: Lift pipette from liquid
4. **Move**: Horizontal movement to target
5. **Down**: Lower pipette to target well
6. **Dispense**: Press button to dispense
7. **Up**: Lift pipette from well
8. **Move**: Return to rest position

**Pre-wetting Technique (12+ direction changes)**:
1. **Down**: Lower pipette to ethanol surface
2. **Pre-wet 1**: Aspirate small volume
3. **Pre-dispense 1**: Dispense back into source (equilibrates tip)
4. **Pre-wet 2**: Second aspiration cycle 
5. **Pre-dispense 2**: Second dispense back into source
6. **Final aspirate**: Aspirate target volume
7. **Up**: Lift pipette from liquid
8. **Move**: Horizontal movement to target
9. **Down**: Lower pipette to target well
10. **Dispense**: Press button to dispense
11. **Up**: Lift pipette from well
12. **Move**: Return to rest position

**Reasoning**: 
- **12 direction changes**: Pre-wetting requires initial aspiration-dispense cycles, doubling normal movement patterns
- **Graduated scoring**: Accounts for partial adherence (some pre-wetting) vs. complete adherence
- **Ethanol volatility**: High vapor pressure (5.95 kPa at 20°C vs water's 2.34 kPa) causes tip dripping without pre-wetting, affecting accuracy
- **Physics basis**: Ethanol vapor expands air cushion in pipette, causing uncontrolled liquid expulsion

### 4. Spatial Movement Metrics

#### 4.1 Spatial Variability
**Formula**: `spatial_variability = std(distances_from_center)`
- Where `distances_from_center = sqrt((x - center_x)² + (y - center_y)²)`
- Units: **pixels** (standard deviation of distances)

**English Explanation**: Measures how much hand position varies from the average working position during protocol execution.

**Distance Calculation**:
- `center_x = mean(all_hand_x_positions)` across entire protocol
- `center_y = mean(all_hand_y_positions)` across entire protocol  
- `distance_i = sqrt((x_i - center_x)² + (y_i - center_y)²)` for each frame i
- `spatial_variability = standard_deviation(all_distances)`

**Reasoning**: 
- **Euclidean distance**: Standard geometric measure of position deviation
- **Center-based**: Uses mean position as reference to account for individual working preferences
- **Lower values**: Indicate more consistent, controlled movements (typical expert: <30 pixels, novice: >80 pixels)
- **Units context**: 30 pixels ≈ 3-8mm real-world spatial consistency depending on camera setup

#### 4.2 Hand Steadiness Score
**Formula**: `steadiness_score = 1.0 - (velocity_std / (velocity_mean + 0.001))`
- Where `velocity = sqrt(dx² + dy²)` for each frame transition
- Units: **dimensionless score** (0-1 scale)

**English Explanation**: Quantifies hand stability during pipetting operations, with higher scores indicating steadier hands.

**Velocity Statistics**:
- `velocity_mean = average(all_frame_velocities)` in pixels/frame
- `velocity_std = standard_deviation(all_frame_velocities)` in pixels/frame
- **Coefficient of Variation**: `CV = velocity_std / velocity_mean` (dimensionless)
- **Steadiness Score**: `1.0 - CV` (inverted so higher = steadier)

**Reasoning**: 
- **CV-based measure**: Relative variability accounts for different movement speeds across operators
- **Epsilon term (+0.001)**: Prevents division by zero in perfectly still moments (rare but possible)
- **Inversion (1.0 -)**: Makes higher values represent better steadiness (0.9+ = very steady, <0.5 = shaky)
- **Biological interpretation**: Reflects natural hand tremor, fatigue, and motor control precision

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

**Component Variability Definitions**:

**Timing Variability**:
- **Formula**: `timing_variability = std(cycle_durations)` 
- **Units**: **seconds** (standard deviation of cycle completion times)
- **Calculation**: Protocol divided into 4 cycles (one per liquid), duration measured for each cycle
- **Example**: If cycle times are [45s, 50s, 48s, 52s], then timing_variability = std([45,50,48,52]) = 2.9 seconds
- **Interpretation**: Lower values indicate more consistent pacing across protocol cycles

**Cycle Duration Context by Liquid Type**:
- **DYE ddH₂O cycle**: Longest duration (8-12 min) due to 12 dispensing events (3× each volume)
- **Other cycles**: Shorter duration (3-7 min) due to 4 dispensing events each
- **Complexity impact**: GLYCEROL and ETHANOL cycles often take longer due to specialized techniques
- **Skill differentiation**: Experts maintain consistent timing regardless of liquid complexity

**Real-world Examples**:
- **Expert operator**: [480s, 240s, 300s, 280s] → timing_variability = 105s (excellent consistency)
- **Trained student**: [600s, 300s, 420s, 380s] → timing_variability = 127s (good consistency)  
- **Novice operator**: [900s, 600s, 750s, 680s] → timing_variability = 123s (surprising consistency despite slower pace)

**Velocity Variability**: 
- **Formula**: `velocity_variability = coefficient_of_variation(cycle_velocity_patterns)`
- **Units**: **dimensionless** (relative variability in movement patterns)
- **Calculation**: 
  - Divide protocol into 4 cycles
  - Calculate mean and std velocity for each cycle
  - Measure variability in these statistics across cycles
  - `velocity_variability = (std_cv + mean_cv) / 2` where cv = coefficient of variation
- **Example**: If cycles show velocity CVs of [0.2, 0.25, 0.22, 0.28], velocity_variability reflects consistency of movement patterns
- **Interpretation**: Lower values indicate more repeatable movement dynamics across cycles

**Reasoning**: 
- **Normalization factors**: 
  - **10.0s timing**: Maximum acceptable cycle-to-cycle timing variation (expert: <2s, novice: >5s)
  - **100px spatial**: Maximum acceptable spatial inconsistency (expert: <30px, novice: >80px) 
  - **50px/frame velocity**: Maximum acceptable velocity pattern variation
- **Equal weighting**: All three components equally important for overall technique assessment
- **0-1 scale**: Provides intuitive scoring system where 1.0 = perfect reproducibility, 0.0 = highly variable

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

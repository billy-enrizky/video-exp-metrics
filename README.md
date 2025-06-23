# Pipetting Video Analysis Framework

# Quick Run
```bash
python reproducibility_experiment.py C0045.MP4 --operator-type experienced_lab_worker --operator-id C0045 --output-dir reproducibility_results
```

## Overview

This framework is designed to analyze videos of people performing liquid handling/pipetting tasks to extract meaningful metrics that can help identify factors contributing to accuracy and consistency differences between operators.

## Key Metrics Extracted

### 1. Hand Movement Analysis
- **Movement Trajectories**: X,Y coordinates of dominant and non-dominant hands over time
- **Velocity Patterns**: Speed of hand movements (pixels/frame)
- **Movement Smoothness**: Standard deviation of velocity (lower = smoother)
- **Pause Detection**: Identification of stationary periods
- **Jerkiness Score**: Measure of abrupt movement changes

### 2. Pipette Handling Metrics
- **Pipette Angle Tracking**: Orientation of pipette relative to vertical
- **Tip Stability**: Consistency of pipette tip position
- **Grip Consistency**: Hand position relative to pipette
- **Aspiration/Dispensing Events**: Detection and timing of liquid operations

### 3. Timing Analysis
- **Task Completion Time**: Total time to complete the procedure
- **Active vs. Pause Time**: Ratio of working time to thinking/pause time
- **Cycle Timing**: Time per pipetting cycle
- **Workflow Efficiency**: Optimization of movement sequences

### 4. Consistency Metrics
- **Spatial Consistency**: Repeatability of hand positions
- **Temporal Consistency**: Repeatability of timing patterns
- **Movement Pattern Consistency**: Similarity across repetitions
- **Overall Reproducibility Score**: Combined consistency measure

## Technical Implementation

### Technologies Used
- **OpenCV**: Video processing and computer vision
- **MediaPipe**: Hand detection and tracking
- **NumPy/Pandas**: Data processing and analysis
- **Matplotlib/Seaborn**: Visualization
- **scikit-learn**: Pattern analysis and clustering

### Analysis Pipeline

1. **Video Preprocessing**
   - Frame extraction and sampling
   - Color space conversion
   - Noise reduction

2. **Hand Detection**
   - MediaPipe-based hand landmark detection
   - Confidence scoring
   - Hand classification (dominant/non-dominant)

3. **Pipette Detection**
   - Color-based segmentation for pipette identification
   - Contour analysis for shape recognition
   - Angle calculation using geometric analysis

4. **Feature Extraction**
   - Movement velocity and acceleration
   - Spatial distribution analysis
   - Temporal pattern recognition

5. **Metric Calculation**
   - Statistical analysis of movement patterns
   - Consistency scoring algorithms
   - Performance benchmarking

## Expected Insights

### Factors Contributing to Higher Accuracy
- **Smoother Hand Movements**: Less jerkiness in pipette handling
- **Consistent Pipette Angles**: Maintaining optimal orientation
- **Stable Tip Position**: Reduced tremor and micro-movements
- **Optimal Timing**: Neither too fast (rushed) nor too slow (inefficient)

### Factors Contributing to Higher Consistency
- **Repeatable Movement Patterns**: Similar trajectories across cycles
- **Consistent Timing**: Regular rhythm and pacing
- **Stable Grip**: Consistent hand position on pipette
- **Predictable Workflow**: Similar sequence of actions

## Potential Applications

### Training and Improvement
- Identify specific areas where operators can improve
- Provide objective feedback on technique
- Track improvement over time

### Quality Control
- Establish benchmarks for acceptable performance
- Identify operators who may need additional training
- Monitor consistency across different sessions

### Research Applications
- Compare different pipetting techniques
- Analyze the effect of fatigue on performance
- Study ergonomic factors affecting accuracy

## Sample Analysis Results

Based on the demo analysis, here are examples of metrics that would be extracted:

```
PERFORMANCE METRICS EXAMPLE:
- Overall Consistency Score: 0.76 (Good)
- Movement Smoothness: 0.73 (Acceptable)  
- Pipette Stability: 0.81 (Very Good)
- Efficiency Score: 0.82 (Very Good)
- Hand Detection Rate: 95% (Excellent)
```

## Recommendations for Implementation

### Video Recording Setup
- **Camera Position**: Side view at hand level for optimal pipette angle visibility
- **Lighting**: Consistent, bright lighting to improve hand detection
- **Background**: Contrasting background to improve object detection
- **Resolution**: Minimum 1080p for adequate detail
- **Frame Rate**: 30 FPS for smooth motion analysis

### File Format Considerations
- Use standard formats (MP4 with H.264 encoding)
- Ensure proper file headers for OpenCV compatibility
- Consider compression vs. quality trade-offs

### Processing Recommendations
- For large files (7GB+), use frame sampling (every 30-60 frames)
- Process in batches to manage memory usage
- Save intermediate results for iterative analysis

## Next Steps

1. **Fix Video Compatibility**: Resolve the current video format issues
2. **Calibration**: Establish baseline metrics from expert operators
3. **Validation**: Compare automated metrics with manual assessments
4. **Scaling**: Process multiple videos for comparative analysis
5. **ML Integration**: Develop predictive models for performance assessment

## Code Files in Framework

- `real_video_analyzer.py`: Main analysis pipeline
- `pipetting_analyzer.py`: Core analysis functions
- `multi_video_analyzer.py`: Batch processing capabilities
- `demo_analyzer.py`: Demonstration with synthetic data
- `quick_test.py`: Basic functionality testing

This framework provides a comprehensive approach to analyzing pipetting technique videos and can be adapted for your specific experimental needs.

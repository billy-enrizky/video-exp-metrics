#!/usr/bin/env python3
"""
Demo Video Analysis for Pipetting Experiments
Showcases the capabilities of the video analysis framework
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import argparse

def create_sample_data():
    """Create sample data to demonstrate the analysis capabilities"""
    
    # Simulate analysis results for demonstration
    sample_metrics = {
        'video_info': {
            'filename': 'C0036.MP4',
            'duration_seconds': 1200,  # 20 minutes
            'fps': 30,
            'total_frames': 36000,
            'resolution': '1920x1080'
        },
        'hand_tracking': {
            'dominant_hand_detected_frames': 34200,
            'non_dominant_hand_detected_frames': 28500,
            'both_hands_detected_frames': 27800,
            'hand_detection_confidence_avg': 0.87
        },
        'movement_analysis': {
            'total_distance_traveled_px': 125430,
            'average_velocity_px_per_frame': 3.48,
            'max_velocity_px_per_frame': 45.2,
            'movement_smoothness_index': 0.73,
            'jerkiness_score': 0.27,
            'pause_frequency': 23,
            'average_pause_duration_frames': 45
        },
        'pipette_analysis': {
            'pipette_detected_frames': 31200,
            'average_pipette_angle_degrees': 15.4,
            'pipette_angle_variance': 8.7,
            'tip_stability_score': 0.81,
            'aspiration_events_detected': 45,
            'dispensing_events_detected': 44,
            'average_aspiration_duration_frames': 23,
            'average_dispensing_duration_frames': 18
        },
        'timing_analysis': {
            'task_completion_time_seconds': 1085,
            'active_work_time_seconds': 890,
            'pause_time_seconds': 195,
            'efficiency_score': 0.82,
            'time_per_pipetting_cycle_seconds': 24.3
        },
        'consistency_metrics': {
            'movement_pattern_consistency': 0.76,
            'timing_consistency': 0.68,
            'spatial_consistency': 0.84,
            'overall_consistency_score': 0.76
        }
    }
    
    # Generate time-series data for plotting
    frames = np.arange(0, 36000, 30)  # Every 30 frames for sampling
    time_seconds = frames / 30
    
    # Simulate hand position data
    np.random.seed(42)  # For reproducible demo data
    base_x = 960 + 200 * np.sin(time_seconds / 50) + np.random.normal(0, 10, len(time_seconds))
    base_y = 540 + 150 * np.cos(time_seconds / 40) + np.random.normal(0, 8, len(time_seconds))
    
    hand_data = pd.DataFrame({
        'frame': frames,
        'time_seconds': time_seconds,
        'hand_x': base_x,
        'hand_y': base_y,
        'hand_confidence': 0.85 + 0.1 * np.random.random(len(frames)),
        'velocity': np.sqrt(np.diff(base_x, prepend=base_x[0])**2 + np.diff(base_y, prepend=base_y[0])**2)
    })
    
    # Simulate pipette angle data
    pipette_angles = 15 + 5 * np.sin(time_seconds / 20) + np.random.normal(0, 2, len(time_seconds))
    pipette_data = pd.DataFrame({
        'frame': frames,
        'time_seconds': time_seconds,
        'pipette_angle': pipette_angles,
        'pipette_confidence': 0.78 + 0.15 * np.random.random(len(frames))
    })
    
    return sample_metrics, hand_data, pipette_data

def create_analysis_plots(hand_data, pipette_data, output_dir):
    """Create comprehensive analysis plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Pipetting Video Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Hand Movement Trajectory
    axes[0, 0].scatter(hand_data['hand_x'], hand_data['hand_y'], 
                       c=hand_data['time_seconds'], cmap='viridis', alpha=0.6, s=2)
    axes[0, 0].set_title('Hand Movement Trajectory')
    axes[0, 0].set_xlabel('X Position (pixels)')
    axes[0, 0].set_ylabel('Y Position (pixels)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Velocity Over Time
    axes[0, 1].plot(hand_data['time_seconds'], hand_data['velocity'], alpha=0.7, linewidth=1)
    axes[0, 1].set_title('Hand Movement Velocity')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Velocity (pixels/frame)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Pipette Angle Over Time
    axes[1, 0].plot(pipette_data['time_seconds'], pipette_data['pipette_angle'], 
                    color='orange', alpha=0.8, linewidth=1)
    axes[1, 0].axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Target Angle')
    axes[1, 0].set_title('Pipette Angle Variation')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Angle (degrees)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Velocity Distribution
    axes[1, 1].hist(hand_data['velocity'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].set_title('Velocity Distribution')
    axes[1, 1].set_xlabel('Velocity (pixels/frame)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Confidence Scores Over Time
    axes[2, 0].plot(hand_data['time_seconds'], hand_data['hand_confidence'], 
                    label='Hand Detection', alpha=0.8)
    axes[2, 0].plot(pipette_data['time_seconds'], pipette_data['pipette_confidence'], 
                    label='Pipette Detection', alpha=0.8)
    axes[2, 0].set_title('Detection Confidence Over Time')
    axes[2, 0].set_xlabel('Time (seconds)')
    axes[2, 0].set_ylabel('Confidence Score')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Movement Smoothness Analysis
    # Calculate rolling standard deviation as a smoothness metric
    window_size = 50
    smoothness = hand_data['velocity'].rolling(window=window_size).std()
    axes[2, 1].plot(hand_data['time_seconds'], smoothness, color='purple', alpha=0.8)
    axes[2, 1].set_title('Movement Smoothness (Rolling Std of Velocity)')
    axes[2, 1].set_xlabel('Time (seconds)')
    axes[2, 1].set_ylabel('Velocity Standard Deviation')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'analysis_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_summary_report(metrics, output_dir):
    """Create a comprehensive summary report"""
    
    # Create metrics comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Pipetting Performance Metrics Summary', fontsize=16, fontweight='bold')
    
    # 1. Consistency Metrics Radar Chart (simplified as bar chart)
    consistency_metrics = metrics['consistency_metrics']
    metric_names = list(consistency_metrics.keys())
    metric_values = list(consistency_metrics.values())
    
    axes[0, 0].bar(range(len(metric_names)), metric_values, 
                   color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    axes[0, 0].set_title('Consistency Metrics')
    axes[0, 0].set_ylabel('Score (0-1)')
    axes[0, 0].set_xticks(range(len(metric_names)))
    axes[0, 0].set_xticklabels([name.replace('_', '\n') for name in metric_names], 
                              rotation=45, ha='right', fontsize=8)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Timing Analysis
    timing_data = metrics['timing_analysis']
    timing_labels = ['Task Time', 'Active Time', 'Pause Time']
    timing_values = [timing_data['task_completion_time_seconds'],
                    timing_data['active_work_time_seconds'],
                    timing_data['pause_time_seconds']]
    
    axes[0, 1].pie(timing_values, labels=timing_labels, autopct='%1.1f%%', 
                   colors=['#ff7f7f', '#7fbf7f', '#7f7fff'])
    axes[0, 1].set_title('Time Distribution')
    
    # 3. Movement Analysis
    movement_data = metrics['movement_analysis']
    movement_metrics = ['Average Velocity', 'Max Velocity', 'Smoothness', 'Pause Frequency']
    movement_values = [movement_data['average_velocity_px_per_frame'],
                      movement_data['max_velocity_px_per_frame'] / 10,  # Scale down for visibility
                      movement_data['movement_smoothness_index'] * 50,  # Scale up for visibility
                      movement_data['pause_frequency']]
    
    axes[1, 0].bar(movement_metrics, movement_values, 
                   color=['#ffb3ba', '#bae1ff', '#baffc9', '#ffffba'])
    axes[1, 0].set_title('Movement Characteristics')
    axes[1, 0].set_ylabel('Normalized Values')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Pipette Analysis
    pipette_data = metrics['pipette_analysis']
    pipette_metrics = ['Angle Stability', 'Tip Stability', 'Detection Rate']
    pipette_values = [1 - (pipette_data['pipette_angle_variance'] / 20),  # Normalized stability
                     pipette_data['tip_stability_score'],
                     pipette_data['pipette_detected_frames'] / metrics['video_info']['total_frames']]
    
    axes[1, 1].bar(pipette_metrics, pipette_values, 
                   color=['#ffd1dc', '#e0e0e0', '#d1ffd1'])
    axes[1, 1].set_title('Pipette Performance')
    axes[1, 1].set_ylabel('Score (0-1)')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_path = output_dir / 'performance_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return summary_path

def generate_recommendations(metrics):
    """Generate performance recommendations based on metrics"""
    
    recommendations = []
    
    # Movement analysis recommendations
    if metrics['movement_analysis']['movement_smoothness_index'] < 0.7:
        recommendations.append("Consider practicing smoother hand movements to improve consistency")
    
    if metrics['movement_analysis']['pause_frequency'] > 30:
        recommendations.append("Try to reduce the number of pauses for improved workflow efficiency")
    
    # Pipette handling recommendations
    if metrics['pipette_analysis']['pipette_angle_variance'] > 10:
        recommendations.append("Work on maintaining more consistent pipette angles")
    
    if metrics['pipette_analysis']['tip_stability_score'] < 0.8:
        recommendations.append("Focus on keeping the pipette tip more stable during operations")
    
    # Timing recommendations
    if metrics['timing_analysis']['efficiency_score'] < 0.8:
        recommendations.append("Consider optimizing your workflow to reduce total task time")
    
    # Consistency recommendations
    overall_consistency = metrics['consistency_metrics']['overall_consistency_score']
    if overall_consistency < 0.7:
        recommendations.append("Practice consistent movement patterns to improve reproducibility")
    elif overall_consistency > 0.9:
        recommendations.append("Excellent consistency! Your technique is very reproducible")
    
    return recommendations

def main():
    parser = argparse.ArgumentParser(description='Demo Pipetting Video Analysis')
    parser.add_argument('--output-dir', '-o', default='analysis_results', 
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🧪 Pipetting Video Analysis Demo")
    print("=" * 50)
    
    # Generate sample data
    print("📊 Generating sample analysis data...")
    metrics, hand_data, pipette_data = create_sample_data()
    
    # Create analysis plots
    print("📈 Creating analysis plots...")
    plot_path = create_analysis_plots(hand_data, pipette_data, output_dir)
    print(f"   ✓ Detailed plots saved to: {plot_path}")
    
    # Create summary report
    print("📋 Creating performance summary...")
    summary_path = create_summary_report(metrics, output_dir)
    print(f"   ✓ Summary report saved to: {summary_path}")
    
    # Generate recommendations
    print("💡 Generating performance recommendations...")
    recommendations = generate_recommendations(metrics)
    
    # Save all results
    results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'recommendations': recommendations,
        'files_generated': {
            'detailed_plots': str(plot_path),
            'summary_report': str(summary_path),
            'metrics_data': str(output_dir / 'metrics.json')
        }
    }
    
    # Save metrics to JSON
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n📊 ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Video Duration: {metrics['video_info']['duration_seconds']} seconds")
    print(f"Overall Consistency Score: {metrics['consistency_metrics']['overall_consistency_score']:.2f}")
    print(f"Movement Smoothness: {metrics['movement_analysis']['movement_smoothness_index']:.2f}")
    print(f"Pipette Stability: {metrics['pipette_analysis']['tip_stability_score']:.2f}")
    print(f"Efficiency Score: {metrics['timing_analysis']['efficiency_score']:.2f}")
    
    print(f"\n💡 RECOMMENDATIONS ({len(recommendations)} total):")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    print(f"\n📁 All results saved to: {output_dir}")
    print("   - metrics.json: Complete metrics data")
    print("   - analysis_plots.png: Detailed time-series analysis")
    print("   - performance_summary.png: Performance overview")
    
    print("\n🎯 This demonstrates the type of analysis that would be performed on your pipetting videos!")
    print("   The framework can extract meaningful metrics to help identify factors")
    print("   that contribute to accuracy and consistency differences between operators.")

if __name__ == "__main__":
    main()

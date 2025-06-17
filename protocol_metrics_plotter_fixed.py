#!/usr/bin/env python3
"""
Protocol Metrics Plotter - Fixed Version
Creates comprehensive visualizations for pipetting protocol analysis metrics
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import warnings
import traceback
warnings.filterwarnings('ignore')

class ProtocolMetricsPlotter:
    def __init__(self, data_path, output_dir="protocol_plots"):
        """Initialize the plotter with analysis data"""
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load analysis data
        self.load_analysis_data()
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 10
        
    def load_analysis_data(self):
        """Load protocol analysis data from JSON file"""
        try:
            with open(self.data_path, 'r') as f:
                self.data = json.load(f)
            print(f"✅ Loaded analysis data from: {self.data_path}")
            
            # Extract key sections - corrected structure
            metrics = self.data.get('metrics', {})
            self.protocol_analysis = metrics.get('protocol_analysis', {})
            self.video_info = metrics.get('video_info', {})
            self.hand_detection = metrics.get('hand_detection', {})
            self.protocol_config = self.data.get('protocol_config', {})
            
            # Debug: Print available data structure
            print(f"📊 Data sections available:")
            print(f"   - protocol_analysis: {'✓' if self.protocol_analysis else '✗'}")
            print(f"   - video_info: {'✓' if self.video_info else '✗'}")
            print(f"   - protocol_config: {'✓' if self.protocol_config else '✗'}")
            print(f"   - hand_detection: {'✓' if self.hand_detection else '✗'}")
            
            # Debug: Show protocol_analysis keys if available
            if self.protocol_analysis:
                print(f"   - protocol_analysis keys: {list(self.protocol_analysis.keys())}")
                if 'reproducibility_analysis' in self.protocol_analysis:
                    repro = self.protocol_analysis['reproducibility_analysis']
                    print(f"   - reproducibility_analysis keys: {list(repro.keys())}")
                    if 'reproducibility_metrics' in repro:
                        metrics = repro['reproducibility_metrics']
                        print(f"   - reproducibility_metrics keys: {list(metrics.keys())}")
            
        except Exception as e:
            raise Exception(f"Failed to load analysis data: {e}")
    
    def create_protocol_overview_plot(self):
        """Create overview plot of protocol execution metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Protocol Execution Overview', fontsize=16, fontweight='bold')
        
        # 1. Protocol Events Summary
        ax1 = axes[0, 0]
        events = self.protocol_analysis.get('protocol_events', {})
        event_types = ['Aspiration', 'Dispensing', 'Tip Changes']
        event_counts = [
            events.get('total_aspiration_events', 0),
            events.get('total_dispensing_events', 0),
            events.get('total_tip_changes', 0)
        ]
        expected_counts = [16, 16, 4]  # From protocol config
        
        x = np.arange(len(event_types))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, event_counts, width, label='Detected', alpha=0.8)
        bars2 = ax1.bar(x + width/2, expected_counts, width, label='Expected', alpha=0.8)
        
        ax1.set_title('Protocol Events: Detected vs Expected')
        ax1.set_xlabel('Event Type')
        ax1.set_ylabel('Count')
        ax1.set_xticks(x)
        ax1.set_xticklabels(event_types)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 2. Timing Metrics
        ax2 = axes[0, 1]
        timing = self.protocol_analysis.get('timing_metrics', {})
        
        timing_labels = ['Total Time\n(min)', 'Avg Cycle\n(sec)', 'Consistency\nScore', 'Efficiency\nScore']
        timing_values = [
            timing.get('total_time_seconds', 0) / 60,
            timing.get('average_cycle_time', 0),
            timing.get('cycle_consistency_score', 0),
            timing.get('time_efficiency_score', 0)
        ]
        
        # Normalize values for display (except total time)
        timing_values_norm = [
            timing_values[0] / 20,  # Normalize to 20 min max
            timing_values[1] / 60,  # Normalize to 60 sec max
            timing_values[2],       # Already 0-1
            min(timing_values[3], 2) / 2  # Normalize to 2 max
        ]
        
        colors = ['blue', 'green', 'orange', 'red']
        bars = ax2.bar(timing_labels, timing_values_norm, color=colors, alpha=0.7)
        ax2.set_title('Timing Performance Metrics')
        ax2.set_ylabel('Normalized Score')
        ax2.set_ylim(0, 1.2)
        ax2.grid(True, alpha=0.3)
        
        # Add actual values as text
        for i, (bar, val) in enumerate(zip(bars, timing_values)):
            if i == 0:
                label = f'{val:.1f} min'
            elif i == 1:
                label = f'{val:.1f} sec'
            else:
                label = f'{val:.3f}'
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    label, ha='center', va='bottom', fontsize=9)
        
        # 3. Skill Assessment
        ax3 = axes[0, 2]
        skill = self.protocol_analysis.get('skill_assessment', {})
        
        skill_score = skill.get('overall_skill_score', 0)
        operator_type = skill.get('operator_type_classification', 'Unknown')
        consistency = skill.get('consistency_rating', 'Unknown')
        
        # Create skill radar-like plot
        categories = ['Skill Score', 'Consistency', 'Speed', 'Accuracy']
        scores = [
            skill_score,
            0.8 if consistency == 'High' else 0.5 if consistency == 'Medium' else 0.2,
            0.8 if skill.get('speed_rating') == 'Optimal' else 0.6 if skill.get('speed_rating') == 'Fast' else 0.4,
            skill_score * 0.9  # Estimated accuracy based on skill
        ]
        
        ax3.bar(categories, scores, color='lightblue', alpha=0.7)
        ax3.set_title(f'Operator Assessment\n({operator_type})')
        ax3.set_ylabel('Score')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Volume Consistency Analysis
        ax4 = axes[1, 0]
        repro_metrics = self.protocol_analysis.get('reproducibility_analysis', {}).get('reproducibility_metrics', {})
        volume_consistency = repro_metrics.get('volume_consistency', {})
        
        if volume_consistency:
            volumes = []
            cv_values = []
            
            for vol_key, vol_data in volume_consistency.items():
                if isinstance(vol_data, dict) and 'volume' in vol_data:
                    volumes.append(vol_data['volume'])
                    cv_values.append(vol_data.get('velocity_consistency_cv', 0))
            
            if volumes and cv_values:
                ax4.bar([f'{v}μL' for v in volumes], cv_values, color='coral', alpha=0.7)
                ax4.set_title('Volume Consistency (CV%)')
                ax4.set_ylabel('Coefficient of Variation (%)')
                ax4.grid(True, alpha=0.3)
                
                # Add target lines
                targets = [5, 3, 2.5, 2]  # Target CV% for each volume
                for i, target in enumerate(targets):
                    if i < len(volumes):
                        ax4.axhline(y=target, color='red', linestyle='--', alpha=0.5)
        else:
            ax4.text(0.5, 0.5, 'No volume\nconsistency data', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Volume Consistency (CV%)')
        
        # 5. Technique Adherence
        ax5 = axes[1, 1]
        technique_adherence = repro_metrics.get('technique_adherence', {})
        
        if technique_adherence:
            techniques = ['Reverse\nPipetting', 'Pre-wetting', 'Contamination\nPrevention']
            scores = [
                technique_adherence.get('reverse_pipetting_glycerol', 0),
                technique_adherence.get('pre_wetting_ethanol', 0),
                technique_adherence.get('cross_contamination_prevention', 0)
            ]
            
            colors = ['green' if s > 0.7 else 'orange' if s > 0.4 else 'red' for s in scores]
            bars = ax5.bar(techniques, scores, color=colors, alpha=0.7)
            ax5.set_title('Technique Adherence Scores')
            ax5.set_ylabel('Adherence Score')
            ax5.set_ylim(0, 1)
            ax5.grid(True, alpha=0.3)
            
            # Add score labels
            for bar, score in zip(bars, scores):
                ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{score:.2f}', ha='center', va='bottom')
        else:
            ax5.text(0.5, 0.5, 'No technique\nadherence data', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Technique Adherence Scores')
        
        # 6. Overall Reproducibility Score
        ax6 = axes[1, 2]
        overall_repro = self.protocol_analysis.get('reproducibility_analysis', {}).get('overall_reproducibility_score', 0)
        
        # Create gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        ax6.plot(x, y, 'k-', linewidth=2)
        ax6.fill_between(x, 0, y, alpha=0.1)
        
        # Color segments
        segments = [(0, 0.3, 'red'), (0.3, 0.7, 'orange'), (0.7, 1.0, 'green')]
        for start, end, color in segments:
            mask = (theta >= start * np.pi) & (theta <= end * np.pi)
            ax6.fill_between(x[mask], 0, y[mask], color=color, alpha=0.3)
        
        # Add needle for current score
        needle_angle = overall_repro * np.pi
        needle_x = [0, 0.8 * np.cos(needle_angle)]
        needle_y = [0, 0.8 * np.sin(needle_angle)]
        ax6.plot(needle_x, needle_y, 'r-', linewidth=4)
        ax6.plot(0, 0, 'ro', markersize=8)
        
        ax6.set_xlim(-1.2, 1.2)
        ax6.set_ylim(-0.2, 1.2)
        ax6.set_aspect('equal')
        ax6.axis('off')
        ax6.set_title(f'Overall Reproducibility\nScore: {overall_repro:.3f}')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'protocol_overview.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Protocol overview plot saved: {plot_path}")
        return plot_path
    
    def create_volume_analysis_plot(self):
        """Create detailed volume-specific analysis plots"""
        repro_metrics = self.protocol_analysis.get('reproducibility_analysis', {}).get('reproducibility_metrics', {})
        volume_consistency = repro_metrics.get('volume_consistency', {})
        
        if not volume_consistency:
            print("⚠️ No volume consistency data available")
            return None
        
        # Extract volume data
        volume_data = []
        for vol_key, vol_data in volume_consistency.items():
            if isinstance(vol_data, dict) and 'cycles' in vol_data:
                for i, cycle in enumerate(vol_data['cycles']):
                    if isinstance(cycle, dict):
                        volume_data.append({
                            'Volume (μL)': vol_data.get('volume', 0),
                            'Cycle': i + 1,
                            'Duration (s)': cycle.get('duration', 0),
                            'Velocity (px/frame)': cycle.get('avg_velocity', 0),
                            'Steadiness Score': cycle.get('steadiness_score', 0),
                            'Total Movement': cycle.get('total_movement', 0)
                        })
        
        if not volume_data:
            print("⚠️ No cycle data available for plotting")
            return None
        
        df = pd.DataFrame(volume_data)
        print(f"📊 Volume analysis data: {len(df)} cycles across volumes")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Volume-Specific Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Duration consistency by volume
        ax1 = axes[0, 0]
        sns.boxplot(data=df, x='Volume (μL)', y='Duration (s)', ax=ax1)
        ax1.set_title('Cycle Duration Consistency by Volume')
        ax1.grid(True, alpha=0.3)
        
        # 2. Velocity patterns by volume
        ax2 = axes[0, 1]
        sns.scatterplot(data=df, x='Volume (μL)', y='Velocity (px/frame)', 
                       size='Total Movement', alpha=0.7, ax=ax2)
        ax2.set_title('Movement Velocity by Volume')
        ax2.grid(True, alpha=0.3)
        
        # 3. Steadiness analysis
        ax3 = axes[1, 0]
        sns.boxplot(data=df, x='Volume (μL)', y='Steadiness Score', ax=ax3)
        ax3.set_title('Hand Steadiness by Volume')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Baseline')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Coefficient of variation summary
        ax4 = axes[1, 1]
        cv_data = []
        for vol_key, vol_data in volume_consistency.items():
            if isinstance(vol_data, dict):
                cv_data.append({
                    'Volume': vol_data.get('volume', 0),
                    'Duration CV': vol_data.get('duration_consistency_cv', 0) * 100,
                    'Velocity CV': vol_data.get('velocity_consistency_cv', 0)
                })
        
        if cv_data:
            cv_df = pd.DataFrame(cv_data)
            x = np.arange(len(cv_df))
            width = 0.35
            
            bars1 = ax4.bar(x - width/2, cv_df['Duration CV'], width, 
                           label='Duration CV (%)', alpha=0.8)
            
            # Scale velocity CV to be comparable
            velocity_cv_scaled = cv_df['Velocity CV'] / 10  # Scale down for visualization
            bars2 = ax4.bar(x + width/2, velocity_cv_scaled, width, 
                           label='Velocity CV (scaled)', alpha=0.8)
            
            ax4.set_title('Coefficient of Variation by Volume')
            ax4.set_xlabel('Volume (μL)')
            ax4.set_ylabel('CV (%)')
            ax4.set_xticks(x)
            ax4.set_xticklabels([f"{int(v)}μL" for v in cv_df['Volume']])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'volume_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Volume analysis plot saved: {plot_path}")
        return plot_path
    
    def create_temporal_analysis_plot(self):
        """Create temporal analysis plots showing performance over time"""
        # Load frame data if available
        frame_data_path = self.data_path.parent / f"protocol_frame_data_{self.data_path.stem.split('_')[-1]}.csv"
        
        if not frame_data_path.exists():
            print(f"⚠️ Frame data not found: {frame_data_path}")
            return None
        
        try:
            df = pd.read_csv(frame_data_path)
            print(f"✅ Loaded frame data: {len(df)} rows")
            print(f"📊 Frame data columns: {list(df.columns)}")
        except Exception as e:
            print(f"❌ Failed to load frame data: {e}")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Hand detection rate over time
        ax1 = axes[0, 0]
        if 'hands_detected' in df.columns and 'time_seconds' in df.columns:
            # Calculate rolling detection rate (percentage of frames with hands detected)
            window_size = min(50, len(df) // 4)
            df['has_hands'] = (df['hands_detected'] > 0).astype(int)
            df['detection_rate'] = df['has_hands'].rolling(window=window_size, center=True).mean()
            
            # Also calculate confident detection rate
            df['confident_detection'] = (df['dom_hand_confidence'] > 0.5).astype(int)
            df['confident_rate'] = df['confident_detection'].rolling(window=window_size, center=True).mean()
            
            ax1.plot(df['time_seconds'] / 60, df['detection_rate'], 
                    label='Any Hand Detection', alpha=0.7, linewidth=2)
            ax1.plot(df['time_seconds'] / 60, df['confident_rate'], 
                    label='Confident Detection (>0.5)', alpha=0.7, linewidth=2)
            
            ax1.set_title('Hand Detection Rate Over Time')
            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('Detection Rate')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.1)
            ax1.legend()
            
            # Add summary statistics
            overall_detection = df['has_hands'].mean()
            overall_confident = df['confident_detection'].mean()
            ax1.text(0.02, 0.98, f'Overall Detection: {overall_detection:.1%}\nConfident Detection: {overall_confident:.1%}', 
                    transform=ax1.transAxes, va='top', ha='left', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax1.text(0.5, 0.5, 'No hand detection\ndata available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Hand Detection Rate Over Time')
        
        # 2. Movement velocity over time
        ax2 = axes[0, 1]
        if 'dominant_hand_velocity' in df.columns:
            # Filter out zero velocities for better visualization
            velocity_data = df['dominant_hand_velocity'].copy()
            velocity_data = velocity_data[velocity_data > 0]  # Remove zero velocities
            
            if len(velocity_data) > 10:
                # Smooth velocity data
                window_size = min(20, len(df) // 10)
                df['velocity_smooth'] = df['dominant_hand_velocity'].rolling(window=window_size, center=True).mean()
                
                # Plot both raw and smoothed data
                ax2.scatter(df['time_seconds'] / 60, df['dominant_hand_velocity'], 
                           alpha=0.3, s=10, color='lightblue', label='Raw Velocity')
                ax2.plot(df['time_seconds'] / 60, df['velocity_smooth'], 
                        color='orange', alpha=0.8, linewidth=2, label='Smoothed')
                
                ax2.set_title('Movement Velocity Over Time')
                ax2.set_xlabel('Time (minutes)')
                ax2.set_ylabel('Velocity (pixels/frame)')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # Add velocity statistics
                mean_vel = velocity_data.mean()
                max_vel = velocity_data.max()
                ax2.text(0.02, 0.98, f'Mean Velocity: {mean_vel:.1f}\nMax Velocity: {max_vel:.1f}', 
                        transform=ax2.transAxes, va='top', ha='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax2.text(0.5, 0.5, 'Insufficient velocity\ndata for plotting', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Movement Velocity Over Time')
        else:
            ax2.text(0.5, 0.5, 'No velocity\ndata available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Movement Velocity Over Time')
        
        # 3. Movement patterns (trajectory)
        ax3 = axes[1, 0]
        if 'dom_hand_x' in df.columns and 'dom_hand_y' in df.columns:
            # Remove NaN values
            valid_data = df.dropna(subset=['dom_hand_x', 'dom_hand_y'])
            
            if len(valid_data) > 100:
                # Color by time
                scatter = ax3.scatter(valid_data['dom_hand_x'], valid_data['dom_hand_y'], 
                                    c=valid_data['time_seconds'], 
                                    cmap='viridis', alpha=0.6, s=20)
                ax3.set_title('Hand Movement Trajectory')
                ax3.set_xlabel('X Position (pixels)')
                ax3.set_ylabel('Y Position (pixels)')
                ax3.grid(True, alpha=0.3)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax3)
                cbar.set_label('Time (seconds)')
            else:
                ax3.text(0.5, 0.5, 'Insufficient trajectory\ndata for plotting', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Hand Movement Trajectory')
        else:
            ax3.text(0.5, 0.5, 'No trajectory\ndata available', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Hand Movement Trajectory')
        
        # 4. Activity level over time
        ax4 = axes[1, 1]
        if 'dominant_hand_velocity' in df.columns:
            # Calculate activity periods
            window_size = min(30, len(df) // 5)
            df['activity_level'] = df['dominant_hand_velocity'].rolling(window=window_size).std()
            
            ax4.plot(df['time_seconds'] / 60, df['activity_level'], 
                    color='red', alpha=0.7, linewidth=2)
            ax4.set_title('Activity Level Over Time')
            ax4.set_xlabel('Time (minutes)')
            ax4.set_ylabel('Movement Variability')
            ax4.grid(True, alpha=0.3)
            
            # Add activity threshold line
            mean_activity = df['activity_level'].mean()
            ax4.axhline(y=mean_activity, color='gray', linestyle='--', alpha=0.5, 
                       label=f'Mean: {mean_activity:.1f}')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No activity\ndata available', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Activity Level Over Time')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'temporal_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Temporal analysis plot saved: {plot_path}")
        return plot_path
    
    def generate_all_plots(self):
        """Generate all available plots"""
        print("🎨 Generating comprehensive protocol metrics plots...")
        print("=" * 60)
        
        plots_created = []
        
        # Generate all plots
        try:
            plot_path = self.create_protocol_overview_plot()
            if plot_path:
                plots_created.append(plot_path)
        except Exception as e:
            print(f"❌ Failed to create protocol overview plot: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            plot_path = self.create_volume_analysis_plot()
            if plot_path:
                plots_created.append(plot_path)
        except Exception as e:
            print(f"❌ Failed to create volume analysis plot: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            plot_path = self.create_temporal_analysis_plot()
            if plot_path:
                plots_created.append(plot_path)
        except Exception as e:
            print(f"❌ Failed to create temporal analysis plot: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n✅ Generated {len(plots_created)} plots:")
        for plot_path in plots_created:
            print(f"   📊 {plot_path}")
        
        return plots_created

def main():
    parser = argparse.ArgumentParser(description='Create protocol metrics plots')
    parser.add_argument('data_path', help='Path to protocol analysis JSON file')
    parser.add_argument('--output-dir', '-o', default='protocol_plots',
                       help='Output directory for plots')
    parser.add_argument('--plot-type', '-t', 
                       choices=['overview', 'volume', 'temporal', 'all'],
                       default='all', help='Type of plot to generate')
    
    args = parser.parse_args()
    
    try:
        # Create plotter
        plotter = ProtocolMetricsPlotter(args.data_path, args.output_dir)
        
        # Generate requested plots
        if args.plot_type == 'overview':
            plotter.create_protocol_overview_plot()
        elif args.plot_type == 'volume':
            plotter.create_volume_analysis_plot()
        elif args.plot_type == 'temporal':
            plotter.create_temporal_analysis_plot()
        else:  # 'all'
            plotter.generate_all_plots()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

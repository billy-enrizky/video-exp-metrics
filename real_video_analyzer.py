#!/usr/bin/env python3
"""
Real Video Analysis for Pipetting Experiments
Processes actual video files to extract meaningful metrics
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
import mediapipe as mp
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class PipettingVideoAnalyzer:
    def __init__(self, video_path, output_dir="real_analysis_results", glove_mode=True, hue_offset=90):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Glove detection parameters
        self.glove_mode = glove_mode
        self.hue_offset = hue_offset  # HSV hue offset for glove detection (0-179)
        
        # Protocol-specific parameters for reproducibility experiment
        self.protocol_config = {
            'volumes': [50, 100, 150, 200],  # μL volumes to dispense
            'liquids': ['DYE_ddH2O', 'DYE-FREE_ddH2O', 'GLYCEROL', 'ETHANOL'],
            'schemes': {
                'DYE_ddH2O': [50, 100, 150, 200],  # All 4 volumes
                'DYE-FREE_ddH2O': [50, 100, 150, 200],  # All 4 volumes  
                'GLYCEROL': [50, 100, 150, 200],  # All 4 volumes
                'ETHANOL': [50, 100, 150, 200]  # All 4 volumes
            },
            'expected_cycles': 16,  # 4 volumes × 4 liquids
            'tip_changes': 4,  # After each liquid type
            'reverse_pipetting_liquids': ['GLYCEROL'],  # Requires reverse pipetting technique
            'pre_wet_liquids': ['ETHANOL'],  # Requires pre-wetting due to volatility
            'operator_types': ['freshly_trained_student', 'experienced_lab_worker', 'automated_handler'],
            'measurement_focus': 'accuracy_and_precision',  # Primary metric of interest
            'expected_duration_minutes': 20,  # Typical protocol duration
            'critical_technique_points': [
                'aspiration_speed',
                'dispensing_angle', 
                'tip_depth_in_liquid',
                'reverse_pipetting_execution',
                'pre_wetting_technique',
                'tip_touching_prevention'
            ]
        }
        
        # Initialize MediaPipe with more sensitive settings for gloved hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,  # Lower threshold for gloves
            min_tracking_confidence=0.3    # Lower threshold for gloves
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Video properties
        self.cap = None
        self.fps = None
        self.frame_count = None
        self.width = None
        self.height = None
        
        # Analysis data
        self.hand_data = []
        self.frame_data = []
        
    def initialize_video(self):
        """Initialize video capture and get properties"""
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video Properties:")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")
        print(f"  Total Frames: {self.frame_count}")
        print(f"  Duration: {self.frame_count / self.fps:.1f} seconds")
        print(f"  File Size: {self.video_path.stat().st_size / (1024**3):.2f} GB")
    
    def apply_hue_offset(self, image, hue_offset):
        """Apply HSV hue offset to improve glove detection"""
        if hue_offset == 0:
            return image
        
        # Convert to HSV
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        
        # Apply hue offset (OpenCV automatically clips to avoid wrap-around)
        h_shifted = cv2.add(h, hue_offset)
        
        # Recombine channels and convert back to BGR
        img_hsv_shifted = cv2.merge([h_shifted, s, v])
        return cv2.cvtColor(img_hsv_shifted, cv2.COLOR_HSV2BGR)
    
    def detect_hands_in_frame(self, frame):
        """Detect hands in a single frame with glove support"""
        if self.glove_mode:
            if self.hue_offset > 0:
                # Method 2: HSV hue offset (more sophisticated)
                processed_frame = self.apply_hue_offset(frame, self.hue_offset)
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            else:
                # Method 1: Simple channel swap (omit color conversion)
                # This makes blue gloves appear orange to MediaPipe
                rgb_frame = frame  # Use BGR directly as RGB
        else:
            # Standard processing for bare hands
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(rgb_frame)
        
        hand_info = {
            'dominant_hand': None,
            'non_dominant_hand': None,
            'hands_detected': 0
        }
        
        if results.multi_hand_landmarks:
            hand_info['hands_detected'] = len(results.multi_hand_landmarks)
            
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Get hand classification (Left/Right)
                hand_label = handedness.classification[0].label
                confidence = handedness.classification[0].score
                
                # Extract landmark positions
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * self.width)
                    y = int(landmark.y * self.height)
                    landmarks.append((x, y))
                
                hand_data = {
                    'landmarks': landmarks,
                    'confidence': confidence,
                    'center': self.calculate_hand_center(landmarks),
                    'bounding_box': self.calculate_bounding_box(landmarks)
                }
                
                # Assign to dominant/non-dominant (assuming right hand is dominant)
                if hand_label == 'Right':
                    hand_info['dominant_hand'] = hand_data
                else:
                    hand_info['non_dominant_hand'] = hand_data
        
        return hand_info
    
    def calculate_hand_center(self, landmarks):
        """Calculate the center point of hand landmarks"""
        if not landmarks:
            return None
        x_coords = [point[0] for point in landmarks]
        y_coords = [point[1] for point in landmarks]
        return (int(np.mean(x_coords)), int(np.mean(y_coords)))
    
    def calculate_bounding_box(self, landmarks):
        """Calculate bounding box of hand landmarks"""
        if not landmarks:
            return None
        x_coords = [point[0] for point in landmarks]
        y_coords = [point[1] for point in landmarks]
        return {
            'x_min': min(x_coords),
            'x_max': max(x_coords),
            'y_min': min(y_coords),
            'y_max': max(y_coords)
        }
    
    def detect_pipette_candidate_regions(self, frame, hand_info):
        """Detect potential pipette regions using simple computer vision"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for typical pipette colors (blue/white/gray)
        # These ranges might need adjustment based on your specific pipettes
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Create masks
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        combined_mask = cv2.bitwise_or(blue_mask, white_mask)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pipette_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 10000:  # Filter by reasonable pipette size
                # Calculate aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h)
                
                if aspect_ratio > 3:  # Pipettes are elongated
                    pipette_candidates.append({
                        'contour': contour,
                        'bounding_box': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })
        
        return pipette_candidates
    
    def analyze_movement_metrics(self, current_hand, previous_hand):
        """Calculate movement metrics between consecutive frames"""
        if not current_hand or not previous_hand:
            return None
        
        current_center = current_hand['center']
        previous_center = previous_hand['center']
        
        if not current_center or not previous_center:
            return None
        
        # Calculate distance and velocity
        dx = current_center[0] - previous_center[0]
        dy = current_center[1] - previous_center[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        return {
            'distance': distance,
            'dx': dx,
            'dy': dy,
            'velocity': distance  # pixels per frame
        }
    
    def process_video_sample(self, max_frames=None, skip_frames=10):
        """Process video with sampling for large files"""
        if not self.cap:
            self.initialize_video()
        
        # For large videos, process every nth frame
        total_frames_to_process = min(max_frames or self.frame_count, self.frame_count)
        frame_indices = list(range(0, total_frames_to_process, skip_frames))
        
        print(f"Processing {len(frame_indices)} frames (every {skip_frames} frames)...")
        
        previous_hand_data = {'dominant': None, 'non_dominant': None}
        
        with tqdm(total=len(frame_indices), desc="Analyzing frames") as pbar:
            for frame_idx in frame_indices:
                # Seek to specific frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # Detect hands
                hand_info = self.detect_hands_in_frame(frame)
                
                # Detect pipette candidates
                pipette_candidates = self.detect_pipette_candidate_regions(frame, hand_info)
                
                # Detect protocol-specific events
                protocol_events = self.detect_protocol_events(hand_info, pipette_candidates, frame_idx)
                
                # Calculate movement metrics
                movement_metrics = {
                    'dominant': None,
                    'non_dominant': None
                }
                
                if hand_info['dominant_hand']:
                    movement_metrics['dominant'] = self.analyze_movement_metrics(
                        hand_info['dominant_hand'], previous_hand_data['dominant']
                    )
                
                if hand_info['non_dominant_hand']:
                    movement_metrics['non_dominant'] = self.analyze_movement_metrics(
                        hand_info['non_dominant_hand'], previous_hand_data['non_dominant']
                    )
                
                # Store frame data with protocol events
                frame_data = {
                    'frame_number': frame_idx,
                    'time_seconds': frame_idx / self.fps,
                    'hands_detected': hand_info['hands_detected'],
                    'dominant_hand': hand_info['dominant_hand'],
                    'non_dominant_hand': hand_info['non_dominant_hand'],
                    'pipette_candidates': len(pipette_candidates),
                    'movement_metrics': movement_metrics,
                    'protocol_events': protocol_events
                }
                
                self.frame_data.append(frame_data)
                
                # Update previous hand data
                previous_hand_data['dominant'] = hand_info['dominant_hand']
                previous_hand_data['non_dominant'] = hand_info['non_dominant_hand']
                
                pbar.update(1)
        
        self.cap.release()
        print(f"✓ Processed {len(self.frame_data)} frames")
    
    def calculate_summary_metrics(self):
        """Calculate summary metrics from processed frame data"""
        if not self.frame_data:
            return {}
        
        # Convert to DataFrame for easier analysis
        df_data = []
        for frame in self.frame_data:
            row = {
                'frame_number': frame['frame_number'],
                'time_seconds': frame['time_seconds'],
                'hands_detected': frame['hands_detected'],
                'pipette_candidates': frame['pipette_candidates']
            }
            
            # Add dominant hand data
            if frame['dominant_hand']:
                row['dom_hand_confidence'] = frame['dominant_hand']['confidence']
                row['dom_hand_x'] = frame['dominant_hand']['center'][0] if frame['dominant_hand']['center'] else None
                row['dom_hand_y'] = frame['dominant_hand']['center'][1] if frame['dominant_hand']['center'] else None
                
                if frame['movement_metrics']['dominant']:
                    row['dom_velocity'] = frame['movement_metrics']['dominant']['velocity']
                    row['dom_distance'] = frame['movement_metrics']['dominant']['distance']
            
            # Add non-dominant hand data
            if frame['non_dominant_hand']:
                row['nondom_hand_confidence'] = frame['non_dominant_hand']['confidence']
                row['nondom_hand_x'] = frame['non_dominant_hand']['center'][0] if frame['non_dominant_hand']['center'] else None
                row['nondom_hand_y'] = frame['non_dominant_hand']['center'][1] if frame['non_dominant_hand']['center'] else None
                
                if frame['movement_metrics']['non_dominant']:
                    row['nondom_velocity'] = frame['movement_metrics']['non_dominant']['velocity']
                    row['nondom_distance'] = frame['movement_metrics']['non_dominant']['distance']
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Calculate summary metrics
        total_frames_processed = len(df)
        hands_detected_frames = len(df[df['hands_detected'] > 0])
        both_hands_frames = len(df[df['hands_detected'] == 2])
        
        # Movement analysis for dominant hand
        dom_velocity_data = df['dom_velocity'].dropna() if 'dom_velocity' in df.columns else pd.Series()
        dom_distance_data = df['dom_distance'].dropna() if 'dom_distance' in df.columns else pd.Series()
        
        metrics = {
            'video_info': {
                'filename': self.video_path.name,
                'duration_seconds': self.frame_count / self.fps if self.fps else 0,
                'fps': self.fps,
                'total_frames': self.frame_count,
                'processed_frames': total_frames_processed,
                'resolution': f"{self.width}x{self.height}",
                'file_size_gb': self.video_path.stat().st_size / (1024**3)
            },
            'hand_detection': {
                'frames_with_hands': hands_detected_frames,
                'frames_with_both_hands': both_hands_frames,
                'hand_detection_rate': hands_detected_frames / total_frames_processed if total_frames_processed > 0 else 0,
                'both_hands_rate': both_hands_frames / total_frames_processed if total_frames_processed > 0 else 0,
                'avg_confidence_dominant': df['dom_hand_confidence'].mean() if 'dom_hand_confidence' in df else 0,
                'avg_confidence_non_dominant': df['nondom_hand_confidence'].mean() if 'nondom_hand_confidence' in df else 0
            },
            'movement_analysis': {
                'total_distance_dominant': dom_distance_data.sum() if not dom_distance_data.empty else 0,
                'avg_velocity_dominant': dom_velocity_data.mean() if not dom_velocity_data.empty else 0,
                'max_velocity_dominant': dom_velocity_data.max() if not dom_velocity_data.empty else 0,
                'velocity_std_dominant': dom_velocity_data.std() if not dom_velocity_data.empty else 0,
                'smoothness_score': 1 / (1 + dom_velocity_data.std()) if not dom_velocity_data.empty and dom_velocity_data.std() > 0 else 0
            },
            'pipette_detection': {
                'avg_pipette_candidates': df['pipette_candidates'].mean(),
                'frames_with_pipette': len(df[df['pipette_candidates'] > 0]),
                'pipette_detection_rate': len(df[df['pipette_candidates'] > 0]) / total_frames_processed if total_frames_processed > 0 else 0
            },
            'timing_analysis': {
                'processing_duration': total_frames_processed / self.fps if self.fps else 0,
                'active_frames': hands_detected_frames,
                'inactive_frames': total_frames_processed - hands_detected_frames,
                'activity_rate': hands_detected_frames / total_frames_processed if total_frames_processed > 0 else 0
            }
        }
        
        return metrics, df
    
    def create_analysis_plots(self, df, metrics):
        """Create comprehensive analysis plots"""
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Real Video Analysis: {self.video_path.name}', fontsize=16, fontweight='bold')
        
        # 1. Hand detection over time
        axes[0, 0].plot(df['time_seconds'], df['hands_detected'], alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Hand Detection Over Time')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Number of Hands Detected')
        axes[0, 0].set_ylim(-0.1, 2.1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Hand movement trajectory (if available)
        if 'dom_hand_x' in df.columns and df['dom_hand_x'].notna().any():
            valid_points = df[df['dom_hand_x'].notna() & df['dom_hand_y'].notna()]
            if not valid_points.empty:
                axes[0, 1].scatter(valid_points['dom_hand_x'], valid_points['dom_hand_y'], 
                                 c=valid_points['time_seconds'], cmap='viridis', alpha=0.6, s=2)
                axes[0, 1].set_title('Dominant Hand Movement Trajectory')
                axes[0, 1].set_xlabel('X Position (pixels)')
                axes[0, 1].set_ylabel('Y Position (pixels)')
                axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No hand position data available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Hand Movement Trajectory')
        
        # 3. Velocity over time (if available)
        if 'dom_velocity' in df.columns and df['dom_velocity'].notna().any():
            velocity_data = df['dom_velocity'].dropna()
            time_data = df.loc[velocity_data.index, 'time_seconds']
            axes[1, 0].plot(time_data, velocity_data, alpha=0.7, linewidth=1, color='orange')
            axes[1, 0].set_title('Dominant Hand Velocity')
            axes[1, 0].set_xlabel('Time (seconds)')
            axes[1, 0].set_ylabel('Velocity (pixels/frame)')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No velocity data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Hand Velocity')
        
        # 4. Detection confidence over time
        confidence_cols = ['dom_hand_confidence', 'nondom_hand_confidence']
        for col in confidence_cols:
            if col in df.columns and df[col].notna().any():
                confidence_data = df[col].dropna()
                time_data = df.loc[confidence_data.index, 'time_seconds']
                label = 'Dominant Hand' if 'dom' in col else 'Non-dominant Hand'
                axes[1, 1].plot(time_data, confidence_data, alpha=0.8, label=label)
        
        axes[1, 1].set_title('Hand Detection Confidence')
        axes[1, 1].set_xlabel('Time (seconds)')
        axes[1, 1].set_ylabel('Confidence Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Pipette detection over time
        axes[2, 0].plot(df['time_seconds'], df['pipette_candidates'], 
                       alpha=0.7, linewidth=1, color='purple')
        axes[2, 0].set_title('Pipette Candidates Over Time')
        axes[2, 0].set_xlabel('Time (seconds)')
        axes[2, 0].set_ylabel('Number of Pipette Candidates')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Activity summary
        activity_data = [
            metrics['hand_detection']['frames_with_hands'],
            metrics['video_info']['processed_frames'] - metrics['hand_detection']['frames_with_hands']
        ]
        labels = ['Active Frames', 'Inactive Frames']
        axes[2, 1].pie(activity_data, labels=labels, autopct='%1.1f%%', 
                      colors=['#66b3ff', '#ff9999'])
        axes[2, 1].set_title('Activity Distribution')
        
        plt.tight_layout()
        plot_path = self.output_dir / f'real_analysis_{self.video_path.stem}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def generate_report(self, metrics, df):
        """Generate a comprehensive analysis report"""
        report_path = self.output_dir / f'analysis_report_{self.video_path.stem}.txt'
        
        with open(report_path, 'w') as f:
            f.write("PIPETTING VIDEO ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Video File: {metrics['video_info']['filename']}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("VIDEO PROPERTIES:\n")
            f.write(f"  Duration: {metrics['video_info']['duration_seconds']:.1f} seconds\n")
            f.write(f"  Resolution: {metrics['video_info']['resolution']}\n")
            f.write(f"  FPS: {metrics['video_info']['fps']}\n")
            f.write(f"  File Size: {metrics['video_info']['file_size_gb']:.2f} GB\n")
            f.write(f"  Frames Processed: {metrics['video_info']['processed_frames']}\n\n")
            
            f.write("HAND DETECTION RESULTS:\n")
            f.write(f"  Hand Detection Rate: {metrics['hand_detection']['hand_detection_rate']:.2%}\n")
            f.write(f"  Both Hands Rate: {metrics['hand_detection']['both_hands_rate']:.2%}\n")
            f.write(f"  Avg Confidence (Dominant): {metrics['hand_detection']['avg_confidence_dominant']:.3f}\n")
            f.write(f"  Avg Confidence (Non-dominant): {metrics['hand_detection']['avg_confidence_non_dominant']:.3f}\n\n")
            
            f.write("MOVEMENT ANALYSIS:\n")
            f.write(f"  Total Distance (Dominant Hand): {metrics['movement_analysis']['total_distance_dominant']:.1f} pixels\n")
            f.write(f"  Average Velocity: {metrics['movement_analysis']['avg_velocity_dominant']:.2f} pixels/frame\n")
            f.write(f"  Max Velocity: {metrics['movement_analysis']['max_velocity_dominant']:.2f} pixels/frame\n")
            f.write(f"  Movement Smoothness Score: {metrics['movement_analysis']['smoothness_score']:.3f}\n\n")
            
            f.write("PIPETTE DETECTION:\n")
            f.write(f"  Average Pipette Candidates: {metrics['pipette_detection']['avg_pipette_candidates']:.1f}\n")
            f.write(f"  Pipette Detection Rate: {metrics['pipette_detection']['pipette_detection_rate']:.2%}\n\n")
            
            f.write("ACTIVITY ANALYSIS:\n")
            f.write(f"  Activity Rate: {metrics['timing_analysis']['activity_rate']:.2%}\n")
            f.write(f"  Active Frames: {metrics['timing_analysis']['active_frames']}\n")
            f.write(f"  Inactive Frames: {metrics['timing_analysis']['inactive_frames']}\n")
        
        return report_path
    
    def run_analysis(self, max_frames=None, skip_frames=30):
        """Run the complete analysis pipeline"""
        print(f"🧪 Starting Real Video Analysis: {self.video_path.name}")
        print("=" * 60)
        
        if self.glove_mode:
            print(f"🧤 Glove detection mode enabled")
            if self.hue_offset > 0:
                print(f"   Using HSV hue offset: {self.hue_offset}")
            else:
                print(f"   Using BGR->RGB channel swap method")
        
        # Initialize video
        self.initialize_video()
        
        # Process video
        print("\n📹 Processing video frames...")
        self.process_video_sample(max_frames=max_frames, skip_frames=skip_frames)
        
        # Calculate metrics
        print("\n📊 Calculating summary metrics...")
        metrics, df = self.calculate_summary_metrics()
        
        # Create plots
        print("\n📈 Creating analysis plots...")
        plot_path = self.create_analysis_plots(df, metrics)
        print(f"   ✓ Plots saved to: {plot_path}")
        
        # Generate report
        print("\n📋 Generating analysis report...")
        report_path = self.generate_report(metrics, df)
        print(f"   ✓ Report saved to: {report_path}")
        
        # Save data
        print("\n💾 Saving analysis data...")
        data_path = self.output_dir / f'analysis_data_{self.video_path.stem}.json'
        with open(data_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            metrics_serializable = json.loads(json.dumps(metrics, default=str))
            json.dump({
                'analysis_timestamp': datetime.now().isoformat(),
                'metrics': metrics_serializable,
                'processing_info': {
                    'max_frames_processed': max_frames,
                    'skip_frames': skip_frames,
                    'total_data_points': len(df)
                }
            }, f, indent=2)
        print(f"   ✓ Data saved to: {data_path}")
        
        # Save DataFrame
        csv_path = self.output_dir / f'frame_data_{self.video_path.stem}.csv'
        df.to_csv(csv_path, index=False)
        print(f"   ✓ Frame data saved to: {csv_path}")
        
        # Print summary
        print(f"\n📊 ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Video Duration: {metrics['video_info']['duration_seconds']:.1f} seconds")
        print(f"Frames Processed: {metrics['video_info']['processed_frames']}")
        print(f"Hand Detection Rate: {metrics['hand_detection']['hand_detection_rate']:.1%}")
        print(f"Movement Smoothness: {metrics['movement_analysis']['smoothness_score']:.3f}")
        print(f"Activity Rate: {metrics['timing_analysis']['activity_rate']:.1%}")
        
        print(f"\n📁 All results saved to: {self.output_dir}")
        
        return metrics, df

    def detect_protocol_events(self, hand_info, pipette_candidates, frame_idx):
        """Detect specific protocol events based on hand and pipette movement patterns"""
        events = {
            'aspiration_event': False,
            'dispensing_event': False,
            'tip_change_event': False,
            'pause_event': False,
            'plate_interaction': False,
            'source_interaction': False
        }
        
        if not hand_info['dominant_hand']:
            return events
        
        hand_center = hand_info['dominant_hand']['center']
        if not hand_center:
            return events
        
        # Analyze movement patterns for event detection
        # High confidence + low movement = dispensing/aspiration
        confidence = hand_info['dominant_hand']['confidence']
        
        # Detect vertical movements (typical for aspiration/dispensing)
        if hasattr(self, 'previous_hand_center') and self.previous_hand_center:
            dy = abs(hand_center[1] - self.previous_hand_center[1])
            dx = abs(hand_center[0] - self.previous_hand_center[0])
            
            # Vertical movement suggests aspiration/dispensing
            if dy > 20 and dy > 2 * dx and confidence > 0.8:
                if hand_center[1] < self.previous_hand_center[1]:  # Moving up
                    events['aspiration_event'] = True
                else:  # Moving down
                    events['dispensing_event'] = True
            
            # Large horizontal movement suggests tip change or moving between containers
            elif dx > 100:
                events['tip_change_event'] = True
            
            # Determine interaction zone based on hand position
            if hand_center[1] < self.height * 0.4:  # Upper part of screen
                events['source_interaction'] = True
            elif hand_center[1] > self.height * 0.6:  # Lower part of screen
                events['plate_interaction'] = True
        
        # Low movement + high confidence = pause/steady operation
        if confidence > 0.85 and pipette_candidates:
            events['pause_event'] = True
        
        self.previous_hand_center = hand_center
        return events
    
    def analyze_protocol_timing(self, frame_data):
        """Analyze timing patterns specific to the protocol"""
        if not frame_data:
            return {}
        
        df = pd.DataFrame([{
            'time_seconds': f['time_seconds'],
            'hands_detected': f['hands_detected'],
            **f.get('protocol_events', {})
        } for f in frame_data])
        
        # Detect pipetting cycles
        aspiration_events = df[df.get('aspiration_event', False)]['time_seconds'].tolist()
        dispensing_events = df[df.get('dispensing_event', False)]['time_seconds'].tolist()
        tip_changes = df[df.get('tip_change_event', False)]['time_seconds'].tolist()
        
        # Calculate cycle timing
        cycle_times = []
        if len(aspiration_events) > 1:
            for i in range(1, len(aspiration_events)):
                cycle_time = aspiration_events[i] - aspiration_events[i-1]
                if 5 < cycle_time < 120:  # Reasonable cycle time range
                    cycle_times.append(cycle_time)
        
        timing_analysis = {
            'total_aspiration_events': len(aspiration_events),
            'total_dispensing_events': len(dispensing_events),
            'total_tip_changes': len(tip_changes),
            'estimated_cycles': min(len(aspiration_events), len(dispensing_events)),
            'average_cycle_time': np.mean(cycle_times) if cycle_times else 0,
            'cycle_time_std': np.std(cycle_times) if cycle_times else 0,
            'cycle_consistency': 1 - (np.std(cycle_times) / np.mean(cycle_times)) if cycle_times and np.mean(cycle_times) > 0 else 0,
            'protocol_completion_estimate': len(aspiration_events) / self.protocol_config['expected_cycles'] if self.protocol_config['expected_cycles'] > 0 else 0
        }
        
        return timing_analysis
    
    def calculate_protocol_accuracy_metrics(self, timing_analysis, movement_data):
        """Calculate accuracy metrics specific to the protocol requirements"""
        
        # Expected vs actual performance
        expected_cycles = self.protocol_config['expected_cycles']
        actual_cycles = timing_analysis.get('estimated_cycles', 0)
        
        # Timing consistency (critical for reproducibility)
        cycle_consistency = timing_analysis.get('cycle_consistency', 0)
        
        # Movement efficiency (smoother = more accurate)
        movement_smoothness = movement_data.get('smoothness_score', 0)
        
        # Protocol adherence score
        protocol_adherence = min(actual_cycles / expected_cycles, 1.0) if expected_cycles > 0 else 0
        
        # Experience level estimation based on multiple factors
        experience_indicators = {
            'timing_consistency': cycle_consistency,
            'movement_smoothness': movement_smoothness,
            'protocol_adherence': protocol_adherence,
            'efficiency': timing_analysis.get('protocol_completion_estimate', 0)
        }
        
        # Overall skill assessment
        skill_score = np.mean(list(experience_indicators.values()))
        
        # Classify experience level
        if skill_score > 0.8:
            experience_level = "Experienced Lab Worker"
        elif skill_score > 0.6:
            experience_level = "Trained Student"
        else:
            experience_level = "Novice/Learning"
        
        return {
            'protocol_accuracy': {
                'expected_cycles': expected_cycles,
                'detected_cycles': actual_cycles,
                'protocol_adherence_score': protocol_adherence,
                'timing_consistency_score': cycle_consistency,
                'movement_quality_score': movement_smoothness,
                'overall_skill_score': skill_score,
                'estimated_experience_level': experience_level
            },
            'experience_indicators': experience_indicators,
            'reproducibility_metrics': {
                'cycle_time_variability': timing_analysis.get('cycle_time_std', 0),
                'expected_variability_expert': 2.0,  # seconds
                'expected_variability_student': 5.0,  # seconds
                'performance_category': 'Expert' if timing_analysis.get('cycle_time_std', 10) < 2.0 else 
                                      'Intermediate' if timing_analysis.get('cycle_time_std', 10) < 5.0 else 'Novice'
            }
        }
    
    def process_video_sample(self, max_frames=None, skip_frames=10):
        """Process video with sampling for large files"""
        if not self.cap:
            self.initialize_video()
        
        # For large videos, process every nth frame
        total_frames_to_process = min(max_frames or self.frame_count, self.frame_count)
        frame_indices = list(range(0, total_frames_to_process, skip_frames))
        
        print(f"Processing {len(frame_indices)} frames (every {skip_frames} frames)...")
        
        previous_hand_data = {'dominant': None, 'non_dominant': None}
        
        with tqdm(total=len(frame_indices), desc="Analyzing frames") as pbar:
            for frame_idx in frame_indices:
                # Seek to specific frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # Detect hands
                hand_info = self.detect_hands_in_frame(frame)
                
                # Detect pipette candidates
                pipette_candidates = self.detect_pipette_candidate_regions(frame, hand_info)
                
                # Calculate movement metrics
                movement_metrics = {
                    'dominant': None,
                    'non_dominant': None
                }
                
                if hand_info['dominant_hand']:
                    movement_metrics['dominant'] = self.analyze_movement_metrics(
                        hand_info['dominant_hand'], previous_hand_data['dominant']
                    )
                
                if hand_info['non_dominant_hand']:
                    movement_metrics['non_dominant'] = self.analyze_movement_metrics(
                        hand_info['non_dominant_hand'], previous_hand_data['non_dominant']
                    )
                
                # Detect protocol-specific events
                protocol_events = self.detect_protocol_events(hand_info, pipette_candidates, frame_idx)
                
                # Store frame data
                frame_data = {
                    'frame_number': frame_idx,
                    'time_seconds': frame_idx / self.fps,
                    'hands_detected': hand_info['hands_detected'],
                    'dominant_hand': hand_info['dominant_hand'],
                    'non_dominant_hand': hand_info['non_dominant_hand'],
                    'pipette_candidates': len(pipette_candidates),
                    'movement_metrics': movement_metrics,
                    'protocol_events': protocol_events
                }
                
                self.frame_data.append(frame_data)
                
                # Update previous hand data
                previous_hand_data['dominant'] = hand_info['dominant_hand']
                previous_hand_data['non_dominant'] = hand_info['non_dominant_hand']
                
                pbar.update(1)
        
        self.cap.release()
        print(f"✓ Processed {len(self.frame_data)} frames")
    
    def calculate_summary_metrics(self):
        """Calculate summary metrics from processed frame data"""
        if not self.frame_data:
            return {}
        
        # Convert to DataFrame for easier analysis
        df_data = []
        for frame in self.frame_data:
            row = {
                'frame_number': frame['frame_number'],
                'time_seconds': frame['time_seconds'],
                'hands_detected': frame['hands_detected'],
                'pipette_candidates': frame['pipette_candidates']
            }
            
            # Add dominant hand data
            if frame['dominant_hand']:
                row['dom_hand_confidence'] = frame['dominant_hand']['confidence']
                row['dom_hand_x'] = frame['dominant_hand']['center'][0] if frame['dominant_hand']['center'] else None
                row['dom_hand_y'] = frame['dominant_hand']['center'][1] if frame['dominant_hand']['center'] else None
                
                if frame['movement_metrics']['dominant']:
                    row['dom_velocity'] = frame['movement_metrics']['dominant']['velocity']
                    row['dom_distance'] = frame['movement_metrics']['dominant']['distance']
            
            # Add non-dominant hand data
            if frame['non_dominant_hand']:
                row['nondom_hand_confidence'] = frame['non_dominant_hand']['confidence']
                row['nondom_hand_x'] = frame['non_dominant_hand']['center'][0] if frame['non_dominant_hand']['center'] else None
                row['nondom_hand_y'] = frame['non_dominant_hand']['center'][1] if frame['non_dominant_hand']['center'] else None
                
                if frame['movement_metrics']['non_dominant']:
                    row['nondom_velocity'] = frame['movement_metrics']['non_dominant']['velocity']
                    row['nondom_distance'] = frame['movement_metrics']['non_dominant']['distance']
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Calculate summary metrics
        total_frames_processed = len(df)
        hands_detected_frames = len(df[df['hands_detected'] > 0])
        both_hands_frames = len(df[df['hands_detected'] == 2])
        
        # Movement analysis for dominant hand
        dom_velocity_data = df['dom_velocity'].dropna() if 'dom_velocity' in df.columns else pd.Series()
        dom_distance_data = df['dom_distance'].dropna() if 'dom_distance' in df.columns else pd.Series()
        
        metrics = {
            'video_info': {
                'filename': self.video_path.name,
                'duration_seconds': self.frame_count / self.fps if self.fps else 0,
                'fps': self.fps,
                'total_frames': self.frame_count,
                'processed_frames': total_frames_processed,
                'resolution': f"{self.width}x{self.height}",
                'file_size_gb': self.video_path.stat().st_size / (1024**3)
            },
            'hand_detection': {
                'frames_with_hands': hands_detected_frames,
                'frames_with_both_hands': both_hands_frames,
                'hand_detection_rate': hands_detected_frames / total_frames_processed if total_frames_processed > 0 else 0,
                'both_hands_rate': both_hands_frames / total_frames_processed if total_frames_processed > 0 else 0,
                'avg_confidence_dominant': df['dom_hand_confidence'].mean() if 'dom_hand_confidence' in df else 0,
                'avg_confidence_non_dominant': df['nondom_hand_confidence'].mean() if 'nondom_hand_confidence' in df else 0
            },
            'movement_analysis': {
                'total_distance_dominant': dom_distance_data.sum() if not dom_distance_data.empty else 0,
                'avg_velocity_dominant': dom_velocity_data.mean() if not dom_velocity_data.empty else 0,
                'max_velocity_dominant': dom_velocity_data.max() if not dom_velocity_data.empty else 0,
                'velocity_std_dominant': dom_velocity_data.std() if not dom_velocity_data.empty else 0,
                'smoothness_score': 1 / (1 + dom_velocity_data.std()) if not dom_velocity_data.empty and dom_velocity_data.std() > 0 else 0
            },
            'pipette_detection': {
                'avg_pipette_candidates': df['pipette_candidates'].mean(),
                'frames_with_pipette': len(df[df['pipette_candidates'] > 0]),
                'pipette_detection_rate': len(df[df['pipette_candidates'] > 0]) / total_frames_processed if total_frames_processed > 0 else 0
            },
            'timing_analysis': {
                'processing_duration': total_frames_processed / self.fps if self.fps else 0,
                'active_frames': hands_detected_frames,
                'inactive_frames': total_frames_processed - hands_detected_frames,
                'activity_rate': hands_detected_frames / total_frames_processed if total_frames_processed > 0 else 0
            }
        }
        
        # Analyze protocol-specific timing
        timing_analysis = self.analyze_protocol_timing(self.frame_data)
        metrics['protocol_timing'] = timing_analysis
        
        # Calculate protocol accuracy metrics
        movement_data = metrics['movement_analysis']
        protocol_accuracy_metrics = self.calculate_protocol_accuracy_metrics(timing_analysis, movement_data)
        metrics['protocol_accuracy'] = protocol_accuracy_metrics
        
        return metrics, df
    
    def create_analysis_plots(self, df, metrics):
        """Create comprehensive analysis plots"""
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(4, 2, figsize=(15, 16))
        fig.suptitle(f'Real Video Analysis: {self.video_path.name}', fontsize=16, fontweight='bold')
        
        # 1. Hand detection over time
        axes[0, 0].plot(df['time_seconds'], df['hands_detected'], alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Hand Detection Over Time')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Number of Hands Detected')
        axes[0, 0].set_ylim(-0.1, 2.1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Hand movement trajectory (if available)
        if 'dom_hand_x' in df.columns and df['dom_hand_x'].notna().any():
            valid_points = df[df['dom_hand_x'].notna() & df['dom_hand_y'].notna()]
            if not valid_points.empty:
                axes[0, 1].scatter(valid_points['dom_hand_x'], valid_points['dom_hand_y'], 
                                 c=valid_points['time_seconds'], cmap='viridis', alpha=0.6, s=2)
                axes[0, 1].set_title('Dominant Hand Movement Trajectory')
                axes[0, 1].set_xlabel('X Position (pixels)')
                axes[0, 1].set_ylabel('Y Position (pixels)')
                axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No hand position data available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Hand Movement Trajectory')
        
        # 3. Velocity over time (if available)
        if 'dom_velocity' in df.columns and df['dom_velocity'].notna().any():
            velocity_data = df['dom_velocity'].dropna()
            time_data = df.loc[velocity_data.index, 'time_seconds']
            axes[1, 0].plot(time_data, velocity_data, alpha=0.7, linewidth=1, color='orange')
            axes[1, 0].set_title('Dominant Hand Velocity')
            axes[1, 0].set_xlabel('Time (seconds)')
            axes[1, 0].set_ylabel('Velocity (pixels/frame)')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No velocity data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Hand Velocity')
        
        # 4. Detection confidence over time
        confidence_cols = ['dom_hand_confidence', 'nondom_hand_confidence']
        for col in confidence_cols:
            if col in df.columns and df[col].notna().any():
                confidence_data = df[col].dropna()
                time_data = df.loc[confidence_data.index, 'time_seconds']
                label = 'Dominant Hand' if 'dom' in col else 'Non-dominant Hand'
                axes[1, 1].plot(time_data, confidence_data, alpha=0.8, label=label)
        
        axes[1, 1].set_title('Hand Detection Confidence')
        axes[1, 1].set_xlabel('Time (seconds)')
        axes[1, 1].set_ylabel('Confidence Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Pipette detection over time
        axes[2, 0].plot(df['time_seconds'], df['pipette_candidates'], 
                       alpha=0.7, linewidth=1, color='purple')
        axes[2, 0].set_title('Pipette Candidates Over Time')
        axes[2, 0].set_xlabel('Time (seconds)')
        axes[2, 0].set_ylabel('Number of Pipette Candidates')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Activity summary
        activity_data = [
            metrics['hand_detection']['frames_with_hands'],
            metrics['video_info']['processed_frames'] - metrics['hand_detection']['frames_with_hands']
        ]
        labels = ['Active Frames', 'Inactive Frames']
        axes[2, 1].pie(activity_data, labels=labels, autopct='%1.1f%%', 
                      colors=['#66b3ff', '#ff9999'])
        axes[2, 1].set_title('Activity Distribution')
        
        # 7. Protocol timing analysis
        timing_metrics = metrics.get('protocol_timing', {})
        axes[3, 0].bar(['Aspiration', 'Dispensing', 'Tip Change'], 
                       [timing_metrics.get('total_aspiration_events', 0), 
                        timing_metrics.get('total_dispensing_events', 0),
                        timing_metrics.get('total_tip_changes', 0)], 
                       color=['#4CAF50', '#2196F3', '#FF9800'])
        axes[3, 0].set_title('Protocol Event Counts')
        axes[3, 0].set_ylabel('Count')
        axes[3, 0].grid(axis='y', alpha=0.3)
        
        if 'average_cycle_time' in timing_metrics:
            axes[3, 1].bar(['Average Cycle Time'], 
                           [timing_metrics['average_cycle_time']], 
                           color=['#9C27B0'])
            axes[3, 1].set_title('Pipetting Cycle Time')
            axes[3, 1].set_ylabel('Time (seconds)')
            axes[3, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / f'real_analysis_{self.video_path.stem}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def generate_report(self, metrics, df):
        """Generate a comprehensive analysis report"""
        report_path = self.output_dir / f'analysis_report_{self.video_path.stem}.txt'
        
        with open(report_path, 'w') as f:
            f.write("PIPETTING VIDEO ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Video File: {metrics['video_info']['filename']}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("VIDEO PROPERTIES:\n")
            f.write(f"  Duration: {metrics['video_info']['duration_seconds']:.1f} seconds\n")
            f.write(f"  Resolution: {metrics['video_info']['resolution']}\n")
            f.write(f"  FPS: {metrics['video_info']['fps']}\n")
            f.write(f"  File Size: {metrics['video_info']['file_size_gb']:.2f} GB\n")
            f.write(f"  Frames Processed: {metrics['video_info']['processed_frames']}\n\n")
            
            f.write("HAND DETECTION RESULTS:\n")
            f.write(f"  Hand Detection Rate: {metrics['hand_detection']['hand_detection_rate']:.2%}\n")
            f.write(f"  Both Hands Rate: {metrics['hand_detection']['both_hands_rate']:.2%}\n")
            f.write(f"  Avg Confidence (Dominant): {metrics['hand_detection']['avg_confidence_dominant']:.3f}\n")
            f.write(f"  Avg Confidence (Non-dominant): {metrics['hand_detection']['avg_confidence_non_dominant']:.3f}\n\n")
            
            f.write("MOVEMENT ANALYSIS:\n")
            f.write(f"  Total Distance (Dominant Hand): {metrics['movement_analysis']['total_distance_dominant']:.1f} pixels\n")
            f.write(f"  Average Velocity: {metrics['movement_analysis']['avg_velocity_dominant']:.2f} pixels/frame\n")
            f.write(f"  Max Velocity: {metrics['movement_analysis']['max_velocity_dominant']:.2f} pixels/frame\n")
            f.write(f"  Movement Smoothness Score: {metrics['movement_analysis']['smoothness_score']:.3f}\n\n")
            
            f.write("PIPETTE DETECTION:\n")
            f.write(f"  Average Pipette Candidates: {metrics['pipette_detection']['avg_pipette_candidates']:.1f}\n")
            f.write(f"  Pipette Detection Rate: {metrics['pipette_detection']['pipette_detection_rate']:.2%}\n\n")
            
            f.write("ACTIVITY ANALYSIS:\n")
            f.write(f"  Activity Rate: {metrics['timing_analysis']['activity_rate']:.2%}\n")
            f.write(f"  Active Frames: {metrics['timing_analysis']['active_frames']}\n")
            f.write(f"  Inactive Frames: {metrics['timing_analysis']['inactive_frames']}\n")
            
            f.write("PROTOCOL TIMING ANALYSIS:\n")
            timing_metrics = metrics.get('protocol_timing', {})
            f.write(f"  Total Aspiration Events: {timing_metrics.get('total_aspiration_events', 0)}\n")
            f.write(f"  Total Dispensing Events: {timing_metrics.get('total_dispensing_events', 0)}\n")
            f.write(f"  Total Tip Changes: {timing_metrics.get('total_tip_changes', 0)}\n")
            f.write(f"  Estimated Cycles: {timing_metrics.get('estimated_cycles', 0)}\n")
            f.write(f"  Average Cycle Time: {timing_metrics.get('average_cycle_time', 0):.1f} seconds\n")
            f.write(f"  Cycle Time Std Dev: {timing_metrics.get('cycle_time_std', 0):.1f} seconds\n")
            f.write(f"  Cycle Consistency: {timing_metrics.get('cycle_consistency', 0):.1%}\n")
            f.write(f"  Protocol Completion Estimate: {timing_metrics.get('protocol_completion_estimate', 0):.1%}\n")
            
            f.write("PROTOCOL ACCURACY METRICS:\n")
            accuracy_metrics = metrics.get('protocol_accuracy', {})
            f.write(f"  Protocol Adherence Score: {accuracy_metrics.get('protocol_adherence', 0):.2%}\n")
            f.write(f"  Timing Consistency Score: {accuracy_metrics.get('timing_consistency', 0):.2%}\n")
            f.write(f"  Movement Quality Score: {accuracy_metrics.get('movement_quality_score', 0):.2%}\n")
            f.write(f"  Overall Skill Score: {accuracy_metrics.get('overall_skill_score', 0):.2%}\n")
            f.write(f"  Estimated Experience Level: {accuracy_metrics.get('estimated_experience_level', 'N/A')}\n")
        
        return report_path
    
    def run_analysis(self, max_frames=None, skip_frames=30):
        """Run the complete analysis pipeline"""
        print(f"🧪 Starting Real Video Analysis: {self.video_path.name}")
        print("=" * 60)
        
        if self.glove_mode:
            print(f"🧤 Glove detection mode enabled")
            if self.hue_offset > 0:
                print(f"   Using HSV hue offset: {self.hue_offset}")
            else:
                print(f"   Using BGR->RGB channel swap method")
        
        # Initialize video
        self.initialize_video()
        
        # Process video
        print("\n📹 Processing video frames...")
        self.process_video_sample(max_frames=max_frames, skip_frames=skip_frames)
        
        # Calculate metrics
        print("\n📊 Calculating summary metrics...")
        metrics, df = self.calculate_summary_metrics()
        
        # Create plots
        print("\n📈 Creating analysis plots...")
        plot_path = self.create_analysis_plots(df, metrics)
        print(f"   ✓ Plots saved to: {plot_path}")
        
        # Generate report
        print("\n📋 Generating analysis report...")
        report_path = self.generate_report(metrics, df)
        print(f"   ✓ Report saved to: {report_path}")
        
        # Save data
        print("\n💾 Saving analysis data...")
        data_path = self.output_dir / f'analysis_data_{self.video_path.stem}.json'
        with open(data_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            metrics_serializable = json.loads(json.dumps(metrics, default=str))
            json.dump({
                'analysis_timestamp': datetime.now().isoformat(),
                'metrics': metrics_serializable,
                'processing_info': {
                    'max_frames_processed': max_frames,
                    'skip_frames': skip_frames,
                    'total_data_points': len(df)
                }
            }, f, indent=2)
        print(f"   ✓ Data saved to: {data_path}")
        
        # Save DataFrame
        csv_path = self.output_dir / f'frame_data_{self.video_path.stem}.csv'
        df.to_csv(csv_path, index=False)
        print(f"   ✓ Frame data saved to: {csv_path}")
        
        # Print summary
        print(f"\n📊 ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Video Duration: {metrics['video_info']['duration_seconds']:.1f} seconds")
        print(f"Frames Processed: {metrics['video_info']['processed_frames']}")
        print(f"Hand Detection Rate: {metrics['hand_detection']['hand_detection_rate']:.1%}")
        print(f"Movement Smoothness: {metrics['movement_analysis']['smoothness_score']:.3f}")
        print(f"Activity Rate: {metrics['timing_analysis']['activity_rate']:.1%}")
        
        print(f"\n📁 All results saved to: {self.output_dir}")
        
        return metrics, df

def main():
    parser = argparse.ArgumentParser(description='Real Pipetting Video Analysis')
    parser.add_argument('video_path', help='Path to the video file to analyze')
    parser.add_argument('--output-dir', '-o', default='real_analysis_results', 
                       help='Output directory for results')
    parser.add_argument('--max-frames', '-m', type=int, default=None,
                       help='Maximum number of frames to process (for testing)')
    parser.add_argument('--skip-frames', '-s', type=int, default=30,
                       help='Process every Nth frame (default: 30)')
    parser.add_argument('--no-gloves', action='store_true',
                       help='Disable glove detection mode (for bare hands)')
    parser.add_argument('--hue-offset', type=int, default=90,
                       help='HSV hue offset for glove detection (0-179, 0=channel swap)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = PipettingVideoAnalyzer(
        args.video_path, 
        args.output_dir,
        glove_mode=not args.no_gloves,
        hue_offset=args.hue_offset
    )
    
    # Run analysis
    try:
        metrics, df = analyzer.run_analysis(
            max_frames=args.max_frames,
            skip_frames=args.skip_frames
        )
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

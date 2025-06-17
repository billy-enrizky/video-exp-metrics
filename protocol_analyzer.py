#!/usr/bin/env python3
"""
Protocol-Specific Pipetting Video Analysis
Analyzes the specific reproducibility experiment protocol with liquid handling tasks
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

class ProtocolPipettingAnalyzer:
    def __init__(self, video_path, output_dir="protocol_analysis_results", glove_mode=True, hue_offset=90):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Glove detection parameters
        self.glove_mode = glove_mode
        self.hue_offset = hue_offset
        
        # Reproducibility Experiment Protocol Configuration
        self.protocol_config = {
            'volumes': [50, 100, 150, 200],  # μL volumes to dispense
            'liquids': ['DYE_ddH2O', 'DYE-FREE_ddH2O', 'GLYCEROL', 'ETHANOL'],
            'schemes': {
                'DYE_ddH2O': [50, 100, 150, 200],      # All 4 volumes with dyed water
                'DYE-FREE_ddH2O': [50, 100, 150, 200], # All 4 volumes with dye-free water
                'GLYCEROL': [50, 100, 150, 200],       # All 4 volumes with glycerol (reverse pipetting)
                'ETHANOL': [50, 100, 150, 200]         # All 4 volumes with ethanol (pre-wetting)
            },
            'expected_cycles': 16,  # 4 volumes × 4 liquids = 16 total dispensing events
            'tip_changes': 4,       # Fresh tip for each liquid type
            'special_techniques': {
                'GLYCEROL': 'reverse_pipetting',  # Higher viscosity requires reverse technique
                'ETHANOL': 'pre_wetting'          # Volatile, requires pre-wetting of tip
            },
            'operator_types': [
                'freshly_trained_student',    # Recently trained, following protocol precisely
                'experienced_lab_worker',     # Years of experience, may have personal variations
                'automated_liquid_handler'    # Mechanical system for comparison baseline
            ],
            'measurement_focus': 'reproducibility_metrics',  # Primary analysis goal
            'accuracy_targets': {
                50: {'tolerance': '±2%', 'expected_cv': '<5%'},
                100: {'tolerance': '±1.5%', 'expected_cv': '<3%'},
                150: {'tolerance': '±1.3%', 'expected_cv': '<2.5%'},
                200: {'tolerance': '±1%', 'expected_cv': '<2%'}
            },
            'critical_consistency_factors': [
                'aspiration_speed_uniformity',
                'dispensing_angle_consistency', 
                'tip_depth_standardization',
                'inter_liquid_contamination_prevention',
                'timing_between_operations',
                'hand_steadiness_throughout_protocol'
            ],
            'expected_completion_time': 1200,  # 20 minutes typical duration
        }
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Video properties
        self.cap = None
        self.fps = None
        self.frame_count = None
        self.width = None
        self.height = None
        
        # Analysis data
        self.frame_data = []
        self.previous_hand_center = None
        
    def initialize_video(self):
        """Initialize video capture and get properties"""
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video Properties:")
        print(f"   Resolution: {self.width}x{self.height}")
        print(f"   FPS: {self.fps}")
        print(f"   Total Frames: {self.frame_count}")
        print(f"   Duration: {self.frame_count / self.fps:.1f} seconds")
        print(f"   File Size: {self.video_path.stat().st_size / (1024**3):.2f} GB")
        
        # Protocol expectations
        print(f"\n🧪 Protocol Expectations:")
        print(f"   Expected Cycles: {self.protocol_config['expected_cycles']}")
        print(f"   Expected Completion: {self.protocol_config['expected_completion_time']/60:.1f} minutes")
        print(f"   Liquids: {', '.join(self.protocol_config['liquids'])}")
        print(f"   Volumes: {self.protocol_config['volumes']} μL")
    
    def apply_hue_offset(self, image, hue_offset):
        """Apply HSV hue offset to improve glove detection"""
        if hue_offset == 0:
            return image
        
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        h_shifted = cv2.add(h, hue_offset)
        img_hsv_shifted = cv2.merge([h_shifted, s, v])
        return cv2.cvtColor(img_hsv_shifted, cv2.COLOR_HSV2BGR)
    
    def detect_hands_in_frame(self, frame):
        """Detect hands with glove support"""
        if self.glove_mode:
            if self.hue_offset > 0:
                processed_frame = self.apply_hue_offset(frame, self.hue_offset)
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame  # BGR->RGB channel swap
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(rgb_frame)
        
        hand_info = {
            'dominant_hand': None,
            'non_dominant_hand': None,
            'hands_detected': 0
        }
        
        if results.multi_hand_landmarks:
            hand_info['hands_detected'] = len(results.multi_hand_landmarks)
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                confidence = handedness.classification[0].score
                
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * self.width)
                    y = int(landmark.y * self.height)
                    landmarks.append((x, y))
                
                hand_data = {
                    'landmarks': landmarks,
                    'confidence': confidence,
                    'center': self.calculate_hand_center(landmarks)
                }
                
                if hand_label == 'Right':
                    hand_info['dominant_hand'] = hand_data
                else:
                    hand_info['non_dominant_hand'] = hand_data
        
        return hand_info
    
    def calculate_hand_center(self, landmarks):
        """Calculate center point of hand landmarks"""
        if not landmarks:
            return None
        x_coords = [point[0] for point in landmarks]
        y_coords = [point[1] for point in landmarks]
        return (int(np.mean(x_coords)), int(np.mean(y_coords)))
    
    def detect_protocol_events(self, hand_info, frame_idx):
        """Detect protocol-specific events"""
        events = {
            'aspiration_event': False,
            'dispensing_event': False,
            'tip_change_event': False,
            'source_interaction': False,
            'plate_interaction': False,
            'steady_operation': False
        }
        
        if not hand_info['dominant_hand'] or not hand_info['dominant_hand']['center']:
            return events
        
        hand_center = hand_info['dominant_hand']['center']
        confidence = hand_info['dominant_hand']['confidence']
        
        # Analyze movement patterns
        if self.previous_hand_center:
            dx = hand_center[0] - self.previous_hand_center[0]
            dy = hand_center[1] - self.previous_hand_center[1]
            movement_magnitude = np.sqrt(dx**2 + dy**2)
            
            # Detect vertical movements (aspiration/dispensing)
            if movement_magnitude > 15 and confidence > 0.8:
                if abs(dy) > 2 * abs(dx):  # Primarily vertical movement
                    if dy < -10:  # Moving up
                        events['aspiration_event'] = True
                    elif dy > 10:  # Moving down
                        events['dispensing_event'] = True
                
                # Large horizontal movement suggests tip change
                elif abs(dx) > 80:
                    events['tip_change_event'] = True
            
            # Determine interaction zones
            if hand_center[1] < self.height * 0.4:  # Upper region
                events['source_interaction'] = True
            elif hand_center[1] > self.height * 0.6:  # Lower region
                events['plate_interaction'] = True
            
            # Steady operation (high confidence, low movement)
            if confidence > 0.85 and movement_magnitude < 5:
                events['steady_operation'] = True
        
        self.previous_hand_center = hand_center
        return events
    
    def analyze_protocol_performance(self, frame_data):
        """Analyze performance metrics specific to the protocol"""
        if not frame_data:
            return {}
        
        # Extract event data
        events_df = pd.DataFrame([{
            'time_seconds': f['time_seconds'],
            'hands_detected': f['hands_detected'],
            **f.get('protocol_events', {})
        } for f in frame_data])
        
        # Count protocol events
        aspiration_events = events_df[events_df.get('aspiration_event', False)]['time_seconds'].tolist()
        dispensing_events = events_df[events_df.get('dispensing_event', False)]['time_seconds'].tolist()
        tip_changes = events_df[events_df.get('tip_change_event', False)]['time_seconds'].tolist()
        
        # Calculate cycle timing
        cycle_times = []
        if len(aspiration_events) > 1:
            for i in range(1, len(aspiration_events)):
                cycle_time = aspiration_events[i] - aspiration_events[i-1]
                if 5 < cycle_time < 180:  # Reasonable cycle time
                    cycle_times.append(cycle_time)
        
        # Performance metrics
        total_time = max(events_df['time_seconds']) if not events_df.empty else 0
        estimated_cycles = min(len(aspiration_events), len(dispensing_events))
        protocol_completion = estimated_cycles / self.protocol_config['expected_cycles'] if self.protocol_config['expected_cycles'] > 0 else 0
        
        # Timing consistency
        cycle_consistency = 1 - (np.std(cycle_times) / np.mean(cycle_times)) if cycle_times and np.mean(cycle_times) > 0 else 0
        
        # Speed assessment
        expected_time = self.protocol_config['expected_completion_time']
        time_efficiency = expected_time / total_time if total_time > 0 else 0
        
        # Experience level estimation
        if cycle_consistency > 0.8 and time_efficiency > 0.8:
            experience_level = "Experienced Lab Worker"
            skill_score = 0.9
        elif cycle_consistency > 0.6 and time_efficiency > 0.6:
            experience_level = "Trained Student"
            skill_score = 0.7
        else:
            experience_level = "Novice/Learning"
            skill_score = 0.4
        
        # Generate comprehensive reproducibility metrics
        # Create proper DataFrame for reproducibility analysis
        df_for_repro = pd.DataFrame([{
            'frame_number': f['frame_number'],
            'time_seconds': f['time_seconds'],
            'hands_detected': f['hands_detected'],
            'dom_hand_confidence': f['dominant_hand']['confidence'] if f['dominant_hand'] else 0,
            'dom_hand_x': f['dominant_hand']['center'][0] if f['dominant_hand'] and f['dominant_hand']['center'] else None,
            'dom_hand_y': f['dominant_hand']['center'][1] if f['dominant_hand'] and f['dominant_hand']['center'] else None,
            'dominant_hand_velocity': f['movement_metrics']['dominant']['velocity'] if f['movement_metrics']['dominant'] else 0,
            'dominant_hand_acceleration': f['movement_metrics']['dominant'].get('acceleration', 0) if f['movement_metrics']['dominant'] else 0
        } for f in self.frame_data])
        
        reproducibility_metrics = self.analyze_reproducibility_metrics(df_for_repro)
        
        # Assess operator type based on comprehensive analysis
        operator_type = self.assess_operator_type(reproducibility_metrics)
        
        # Calculate accuracy predictions
        accuracy_predictions = self.calculate_accuracy_predictions(reproducibility_metrics)
        
        return {
            'protocol_events': {
                'total_aspiration_events': len(aspiration_events),
                'total_dispensing_events': len(dispensing_events),
                'total_tip_changes': len(tip_changes),
                'estimated_cycles': estimated_cycles,
                'protocol_completion_percentage': protocol_completion * 100
            },
            'timing_metrics': {
                'total_time_seconds': total_time,
                'average_cycle_time': np.mean(cycle_times) if cycle_times else 0,
                'cycle_time_std': np.std(cycle_times) if cycle_times else 0,
                'cycle_consistency_score': cycle_consistency,
                'time_efficiency_score': time_efficiency,
                'expected_vs_actual_time_ratio': expected_time / total_time if total_time > 0 else 0
            },
            'skill_assessment': {
                'estimated_experience_level': experience_level,
                'operator_type_classification': operator_type,
                'overall_skill_score': skill_score,
                'consistency_rating': 'High' if cycle_consistency > 0.8 else 'Medium' if cycle_consistency > 0.6 else 'Low',
                'speed_rating': 'Fast' if time_efficiency > 1.2 else 'Optimal' if time_efficiency > 0.8 else 'Slow'
            },
            'reproducibility_analysis': {
                'cycle_variability_seconds': np.std(cycle_times) if cycle_times else 0,
                'reproducibility_metrics': reproducibility_metrics,
                'accuracy_predictions': accuracy_predictions,
                'overall_reproducibility_score': reproducibility_metrics.get('inter_cycle_variability', {}).get('overall_reproducibility_score', 0.0),
                'expected_expert_variability': 2.0,  # seconds
                'expected_student_variability': 5.0,  # seconds
                'performance_category': ('Expert' if (cycle_times and np.std(cycle_times) < 2.0) else
                                      'Intermediate' if (cycle_times and np.std(cycle_times) < 5.0) else 'Novice'),
                'accuracy_prediction': 'High' if cycle_consistency > 0.8 else 'Medium' if cycle_consistency > 0.6 else 'Low'
            }
        }
    
    def process_video(self, max_frames=None, skip_frames=30):
        """Process video with protocol-specific analysis"""
        if not self.cap:
            self.initialize_video()
        
        total_frames_to_process = min(max_frames or self.frame_count, self.frame_count)
        frame_indices = list(range(0, total_frames_to_process, skip_frames))
        
        print(f"\n🔍 Processing {len(frame_indices)} frames (every {skip_frames} frames)...")
        
        previous_hand_data = {'dominant': None, 'non_dominant': None}
        
        with tqdm(total=len(frame_indices), desc="Analyzing protocol execution") as pbar:
            for frame_idx in frame_indices:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # Detect hands
                hand_info = self.detect_hands_in_frame(frame)
                
                # Detect protocol events
                protocol_events = self.detect_protocol_events(hand_info, frame_idx)
                
                # Calculate movement metrics
                movement_metrics = {'dominant': None, 'non_dominant': None}
                if hand_info['dominant_hand'] and previous_hand_data['dominant']:
                    current_center = hand_info['dominant_hand']['center']
                    previous_center = previous_hand_data['dominant']['center'] if previous_hand_data['dominant'] else None
                    
                    if current_center and previous_center:
                        dx = current_center[0] - previous_center[0]
                        dy = current_center[1] - previous_center[1]
                        distance = np.sqrt(dx**2 + dy**2)
                        movement_metrics['dominant'] = {
                            'distance': distance,
                            'velocity': distance,
                            'acceleration': 0  # Simplified for now
                        }
                
                # Store frame data
                frame_data = {
                    'frame_number': frame_idx,
                    'time_seconds': frame_idx / self.fps if self.fps else frame_idx,
                    'hands_detected': hand_info['hands_detected'],
                    'dominant_hand': hand_info['dominant_hand'],
                    'non_dominant_hand': hand_info['non_dominant_hand'],
                    'protocol_events': protocol_events,
                    'movement_metrics': movement_metrics
                }
                
                self.frame_data.append(frame_data)
                
                # Update previous hand data
                previous_hand_data['dominant'] = hand_info['dominant_hand']
                previous_hand_data['non_dominant'] = hand_info['non_dominant_hand']
                
                pbar.update(1)
        
        self.cap.release()
        print(f"✓ Processed {len(self.frame_data)} frames")
    
    def generate_protocol_report(self, metrics):
        """Generate comprehensive protocol analysis report"""
        report_path = self.output_dir / f'protocol_analysis_report_{self.video_path.stem}.txt'
        
        with open(report_path, 'w') as f:
            f.write("PIPETTING PROTOCOL ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Video File: {metrics['video_info']['filename']}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Protocol: Reproducibility Experiment (4 liquids, 4 volumes)\n\n")
            
            # Protocol Performance
            protocol = metrics['protocol_analysis']
            f.write("PROTOCOL EXECUTION SUMMARY:\n")
            f.write(f"  Estimated Cycles Completed: {protocol['protocol_events']['estimated_cycles']}/{self.protocol_config['expected_cycles']}\n")
            f.write(f"  Protocol Completion: {protocol['protocol_events']['protocol_completion_percentage']:.1f}%\n")
            f.write(f"  Total Time: {protocol['timing_metrics']['total_time_seconds']:.1f} seconds ({protocol['timing_metrics']['total_time_seconds']/60:.1f} minutes)\n")
            f.write(f"  Average Cycle Time: {protocol['timing_metrics']['average_cycle_time']:.1f} seconds\n")
            f.write(f"  Cycle Consistency Score: {protocol['timing_metrics']['cycle_consistency_score']:.3f}\n\n")
            
            # Skill Assessment
            skill = protocol['skill_assessment']
            f.write("OPERATOR SKILL ASSESSMENT:\n")
            f.write(f"  Estimated Experience Level: {skill['estimated_experience_level']}\n")
            f.write(f"  Operator Type Classification: {skill['operator_type_classification']}\n")
            f.write(f"  Overall Skill Score: {skill['overall_skill_score']:.2f}/1.0\n")
            f.write(f"  Consistency Rating: {skill['consistency_rating']}\n")
            f.write(f"  Speed Rating: {skill['speed_rating']}\n\n")
            
            # Reproducibility Analysis
            repro = protocol['reproducibility_analysis']
            f.write("REPRODUCIBILITY ANALYSIS:\n")
            f.write(f"  Overall Reproducibility Score: {repro['overall_reproducibility_score']:.3f}/1.0\n")
            f.write(f"  Cycle Time Variability: {repro['cycle_variability_seconds']:.1f} seconds\n")
            
            # Accuracy Predictions
            if 'accuracy_predictions' in repro:
                f.write("\nACCURACY PREDICTIONS BY VOLUME:\n")
                for volume, pred in repro['accuracy_predictions'].items():
                    f.write(f"  {volume}: Predicted CV {pred['predicted_cv_percent']:.1f}% (Target: <{pred['target_cv_percent']:.1f}%) - {'✓ MEETS TARGET' if pred['meets_target'] else '✗ EXCEEDS TARGET'}\n")
            
            # Technique Adherence
            if 'reproducibility_metrics' in repro:
                tech = repro['reproducibility_metrics'].get('technique_adherence', {})
                f.write(f"\nTECHNIQUE ADHERENCE:\n")
                f.write(f"  Reverse Pipetting (Glycerol): {tech.get('reverse_pipetting_glycerol', 0):.2f}/1.0\n")
                f.write(f"  Pre-wetting (Ethanol): {tech.get('pre_wetting_ethanol', 0):.2f}/1.0\n")
                f.write(f"  Tip Change Execution: {tech.get('tip_change_execution', 0):.2f}/1.0\n")
            
            f.write(f"\nComparison to Expert Std ({repro['expected_expert_variability']}s): {'Better' if repro['cycle_variability_seconds'] < repro['expected_expert_variability'] else 'Similar' if repro['cycle_variability_seconds'] < repro['expected_expert_variability'] * 1.5 else 'Needs Improvement'}\n\n")
            
            # Hand Detection Performance
            hand = metrics['hand_detection']
            f.write("TECHNICAL ANALYSIS:\n")
            f.write(f"  Hand Detection Rate: {hand['hand_detection_rate']:.1%}\n")
            f.write(f"  Average Detection Confidence: {hand['avg_confidence_dominant']:.3f}\n")
            f.write(f"  Frames Processed: {metrics['video_info']['processed_frames']}\n")
            
        return report_path
    
    def run_analysis(self, max_frames=None, skip_frames=30):
        """Run complete protocol analysis"""
        print(f"🧪 Starting Protocol-Specific Analysis: {self.video_path.name}")
        print("=" * 70)
        
        if self.glove_mode:
            print(f"🧤 Glove detection enabled (hue offset: {self.hue_offset})")
        
        # Initialize and process video
        self.initialize_video()
        self.process_video(max_frames=max_frames, skip_frames=skip_frames)
        
        # Analyze protocol performance
        print("\n📊 Analyzing protocol performance...")
        protocol_analysis = self.analyze_protocol_performance(self.frame_data)
        
        # Calculate basic metrics
        df_data = []
        for frame in self.frame_data:
            row = {
                'frame_number': frame['frame_number'],
                'time_seconds': frame['time_seconds'],
                'hands_detected': frame['hands_detected']
            }
            
            if frame['dominant_hand']:
                row['dom_hand_confidence'] = frame['dominant_hand']['confidence']
                row['dom_hand_x'] = frame['dominant_hand']['center'][0] if frame['dominant_hand']['center'] else None
                row['dom_hand_y'] = frame['dominant_hand']['center'][1] if frame['dominant_hand']['center'] else None
                
                if frame['movement_metrics']['dominant']:
                    row['dominant_hand_velocity'] = frame['movement_metrics']['dominant']['velocity']
                    row['dominant_hand_acceleration'] = frame['movement_metrics']['dominant'].get('acceleration', 0)
                else:
                    row['dominant_hand_velocity'] = 0
                    row['dominant_hand_acceleration'] = 0
            else:
                row['dom_hand_confidence'] = 0
                row['dom_hand_x'] = None
                row['dom_hand_y'] = None
                row['dominant_hand_velocity'] = 0
                row['dominant_hand_acceleration'] = 0
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Compile final metrics
        total_frames = len(df)
        hands_detected_frames = len(df[df['hands_detected'] > 0])
        
        metrics = {
            'video_info': {
                'filename': self.video_path.name,
                'duration_seconds': self.frame_count / self.fps if self.fps else 0,
                'fps': self.fps,
                'total_frames': self.frame_count,
                'processed_frames': total_frames,
                'resolution': f"{self.width}x{self.height}",
                'file_size_gb': self.video_path.stat().st_size / (1024**3)
            },
            'hand_detection': {
                'frames_with_hands': hands_detected_frames,
                'hand_detection_rate': hands_detected_frames / total_frames if total_frames > 0 else 0,
                'avg_confidence_dominant': df['dom_hand_confidence'].mean() if 'dom_hand_confidence' in df else 0
            },
            'protocol_analysis': protocol_analysis
        }
        
        # Generate report
        print("\n📋 Generating protocol analysis report...")
        report_path = self.generate_protocol_report(metrics)
        
        # Save data
        print("\n💾 Saving analysis data...")
        data_path = self.output_dir / f'protocol_analysis_data_{self.video_path.stem}.json'
        with open(data_path, 'w') as f:
            json.dump({
                'analysis_timestamp': datetime.now().isoformat(),
                'protocol_config': self.protocol_config,
                'metrics': metrics,
                'frame_data_summary': f"{len(self.frame_data)} frames analyzed"
            }, f, indent=2, default=str)
        
        # Save frame data
        csv_path = self.output_dir / f'protocol_frame_data_{self.video_path.stem}.csv'
        df.to_csv(csv_path, index=False)
        
        # Print summary
        print(f"\n📊 PROTOCOL ANALYSIS SUMMARY")
        print("=" * 70)
        protocol = protocol_analysis
        print(f"Estimated Experience Level: {protocol['skill_assessment']['estimated_experience_level']}")
        print(f"Protocol Completion: {protocol['protocol_events']['protocol_completion_percentage']:.1f}%")
        print(f"Cycles Detected: {protocol['protocol_events']['estimated_cycles']}/{self.protocol_config['expected_cycles']}")
        print(f"Cycle Consistency: {protocol['timing_metrics']['cycle_consistency_score']:.3f}")
        print(f"Average Cycle Time: {protocol['timing_metrics']['average_cycle_time']:.1f} seconds")
        print(f"Performance Category: {protocol['reproducibility_analysis']['performance_category']}")
        print(f"Predicted Accuracy: {protocol['reproducibility_analysis']['accuracy_prediction']}")
        
        print(f"\n📁 Results saved to: {self.output_dir}")
        print(f"   - {report_path.name}: Detailed analysis report")
        print(f"   - {data_path.name}: Complete metrics data")
        print(f"   - {csv_path.name}: Frame-by-frame data")
        
        return metrics

    def analyze_reproducibility_metrics(self, df):
        """Analyze reproducibility metrics for the protocol"""
        if df.empty:
            return {'error': 'No data available for analysis'}
        
        # Initialize metrics structure
        metrics = {
            'volume_consistency': {},
            'liquid_consistency': {},
            'technique_adherence': {},
            'inter_cycle_variability': {},
            'overall_metrics': {}
        }
        
        # Analyze consistency for each volume
        for volume in self.protocol_config['volumes']:
            volume_cycles = self._identify_volume_cycles(df, volume)
            metrics['volume_consistency'][f'{volume}uL'] = volume_cycles
        
        # Analyze liquid-specific consistency
        for liquid in self.protocol_config['liquids']:
            liquid_data = self._extract_liquid_specific_data(df, liquid)
            metrics['liquid_consistency'][liquid] = liquid_data
        
        # Assess technique adherence
        metrics['technique_adherence'] = {
            'reverse_pipetting_glycerol': self._assess_reverse_pipetting_technique(df),
            'pre_wetting_ethanol': self._assess_pre_wetting_technique(df),
            'tip_change_technique': self._assess_tip_change_technique(df),
            'cross_contamination_prevention': self._assess_contamination_prevention_overall(df)
        }
        
        # Calculate inter-cycle variability
        metrics['inter_cycle_variability'] = {
            'timing_variability': self._calculate_timing_variability(df),
            'spatial_variability': self._calculate_spatial_variability(df),
            'velocity_variability': self._calculate_velocity_variability(df),
            'overall_reproducibility_score': 0.0  # Will be calculated below
        }
        
        # Calculate overall reproducibility score
        timing_score = 1.0 - min(metrics['inter_cycle_variability']['timing_variability'] / 10.0, 1.0)
        spatial_score = 1.0 - min(metrics['inter_cycle_variability']['spatial_variability'] / 100.0, 1.0)
        velocity_score = 1.0 - min(metrics['inter_cycle_variability']['velocity_variability'] / 50.0, 1.0)
        
        metrics['inter_cycle_variability']['overall_reproducibility_score'] = np.mean([
            timing_score, spatial_score, velocity_score
        ])
        
        return metrics
    
    def _identify_volume_cycles(self, df, target_volume):
        """Identify and analyze cycles for a specific volume"""
        if df.empty:
            return {
                'volume': target_volume,
                'cycles_detected': 0,
                'duration_consistency_cv': 100.0,
                'velocity_consistency_cv': 100.0,
                'average_duration': 0,
                'average_velocity': 0,
                'cycles': []
            }
        
        # Estimate 4 cycles per volume (one for each liquid)
        estimated_cycles_per_volume = 4
        cycles = []
        
        # Divide the data into equal segments for each expected cycle
        for i in range(estimated_cycles_per_volume):
            cycle_start = i * len(df) // estimated_cycles_per_volume
            cycle_end = (i + 1) * len(df) // estimated_cycles_per_volume
            cycle_data = df.iloc[cycle_start:cycle_end]
            
            if not cycle_data.empty and 'dominant_hand_velocity' in cycle_data.columns:
                velocity_data = cycle_data['dominant_hand_velocity'].dropna()
                if len(velocity_data) > 0:
                    cycles.append({
                        'volume': target_volume,
                        'duration': (cycle_data['time_seconds'].iloc[-1] - cycle_data['time_seconds'].iloc[0]) if len(cycle_data) > 1 else 0,
                        'total_movement': velocity_data.sum(),
                        'avg_velocity': velocity_data.mean(),
                        'velocity_std': velocity_data.std(),
                        'steadiness_score': 1.0 - (velocity_data.std() / (velocity_data.mean() + 0.001))
                    })
        
        # Calculate consistency metrics
        if cycles:
            durations = [c['duration'] for c in cycles]
            velocities = [c['avg_velocity'] for c in cycles]
            
            return {
                'volume': target_volume,
                'cycles_detected': len(cycles),
                'duration_consistency_cv': (np.std(durations) / (np.mean(durations) + 0.001)) * 100,
                'velocity_consistency_cv': (np.std(velocities) / (np.mean(velocities) + 0.001)) * 100,
                'average_duration': np.mean(durations),
                'average_velocity': np.mean(velocities),
                'cycles': cycles
            }
        else:
            return {
                'volume': target_volume,
                'cycles_detected': 0,
                'duration_consistency_cv': 100.0,
                'velocity_consistency_cv': 100.0,
                'average_duration': 0,
                'average_velocity': 0,
                'cycles': []
            }
    
    def _extract_liquid_specific_data(self, df, liquid):
        """Extract data segments corresponding to specific liquid handling"""
        # Placeholder - would identify liquid-specific segments based on protocol timing
        liquid_index = self.protocol_config['liquids'].index(liquid)
        segment_size = len(df) // len(self.protocol_config['liquids'])
        start_idx = liquid_index * segment_size
        end_idx = (liquid_index + 1) * segment_size
        
        return df.iloc[start_idx:end_idx]
    
    def _assess_technique_adherence(self, liquid_data, technique_type):
        """Assess adherence to required technique for specific liquids"""
        if technique_type == 'reverse_pipetting':
            # Assess if reverse pipetting pattern is detected
            return self._detect_reverse_pipetting_pattern(liquid_data)
        elif technique_type == 'pre_wetting':
            # Assess if pre-wetting steps are detected
            return self._detect_pre_wetting_pattern(liquid_data)
        else:
            return 1.0  # Standard technique, assume good adherence
    
    def _detect_reverse_pipetting_pattern(self, data):
        """Detect if reverse pipetting technique is being used"""
        # Look for characteristic double-aspiration pattern
        # This would need actual movement analysis - placeholder implementation
        if data.empty:
            return 0.5
        
        velocity_peaks = self._find_velocity_peaks(data)
        # Reverse pipetting should show extra aspiration steps
        expected_extra_steps = 2  # Additional aspirations for reverse technique
        detected_extra_steps = max(0, len(velocity_peaks) - 4)  # Standard has ~4 major movements
        
        adherence_score = min(1.0, detected_extra_steps / expected_extra_steps)
        return adherence_score
    
    def _detect_pre_wetting_pattern(self, data):
        """Detect if pre-wetting technique is being used"""
        # Look for initial aspiration-dispense cycles before actual sampling
        if data.empty:
            return 0.5
        
        velocity_changes = np.diff(data['dominant_hand_velocity'].fillna(0))
        direction_changes = np.sum(np.diff(np.sign(velocity_changes)) != 0)
        
        # Pre-wetting should show extra movement cycles at the beginning
        expected_direction_changes = 12  # More cycles due to pre-wetting
        standard_direction_changes = 8   # Standard technique
        
        if direction_changes >= expected_direction_changes:
            return 1.0
        elif direction_changes >= standard_direction_changes:
            return 0.7
        else:
            return 0.3
    
    def _find_velocity_peaks(self, data):
        """Find velocity peaks in movement data"""
        if 'dominant_hand_velocity' not in data.columns or data['dominant_hand_velocity'].isna().all():
            return []
        
        velocity = data['dominant_hand_velocity'].fillna(0)
        # Simple peak detection
        peaks = []
        for i in range(1, len(velocity) - 1):
            if velocity.iloc[i] > velocity.iloc[i-1] and velocity.iloc[i] > velocity.iloc[i+1]:
                if velocity.iloc[i] > velocity.mean() + velocity.std():
                    peaks.append(i)
        
        return peaks
    
    def assess_operator_type(self, metrics):
        """Assess if operator is freshly trained student, experienced worker, or automated"""
        reproducibility = metrics.get('inter_cycle_variability', {}).get('overall_reproducibility_score', 0.5)
        timing_consistency = metrics.get('timing_metrics', {}).get('cycle_consistency_score', 0.5)
        technique_adherence = np.mean([
            metrics.get('technique_adherence', {}).get('reverse_pipetting_glycerol', 0.5),
            metrics.get('technique_adherence', {}).get('pre_wetting_ethanol', 0.5)
        ])
        
        # Decision logic based on experimental expectations
        if reproducibility > 0.9 and timing_consistency > 0.9:
            return "automated_liquid_handler"
        elif technique_adherence > 0.8 and timing_consistency > 0.7:
            return "experienced_lab_worker"
        elif technique_adherence > 0.6 and reproducibility > 0.5:
            return "freshly_trained_student"
        else:
            return "novice_learning"
    
    def calculate_accuracy_predictions(self, metrics):
        """Predict accuracy based on observed technique metrics"""
        predictions = {}
        
        for volume in self.protocol_config['volumes']:
            volume_key = f'{volume}uL'
            consistency = metrics.get('volume_consistency', {}).get(volume_key, {})
            
            # Predict CV% based on movement consistency
            movement_cv = consistency.get('movement_cv', 0.2)
            timing_cv = consistency.get('timing_cv', 0.2)
            steadiness_cv = consistency.get('hand_steadiness_cv', 0.2)
            
            # Empirical model for CV prediction (would be calibrated with actual data)
            predicted_cv = (movement_cv * 0.4 + timing_cv * 0.3 + steadiness_cv * 0.3) * 100
            
            # Accuracy target from protocol config
            target = self.protocol_config['accuracy_targets'][volume]
            target_cv = float(target['expected_cv'].replace('<', '').replace('%', ''))
            
            predictions[volume_key] = {
                'predicted_cv_percent': predicted_cv,
                'target_cv_percent': target_cv,
                'meets_target': predicted_cv <= target_cv,
                'accuracy_score': max(0.0, 1.0 - (predicted_cv / target_cv))
            }
        
        return predictions

    def _calculate_liquid_consistency(self, liquid_data):
        """Calculate consistency metrics across all volumes for a specific liquid"""
        if liquid_data.empty:
            return 0.5
        
        # Analyze velocity patterns across the liquid handling period
        velocity_data = liquid_data['dominant_hand_velocity'].fillna(0)
        velocity_segments = np.array_split(velocity_data, 4)  # 4 volumes
        
        segment_means = [seg.mean() for seg in velocity_segments]
        segment_stds = [seg.std() for seg in velocity_segments]
        
        # Consistency is inverse of coefficient of variation
        mean_cv = np.std(segment_means) / (np.mean(segment_means) + 0.001)
        std_cv = np.std(segment_stds) / (np.mean(segment_stds) + 0.001)
        
        consistency_score = 1.0 - (mean_cv + std_cv) / 2.0
        return max(0.0, min(1.0, consistency_score))
    
    def _assess_contamination_prevention(self, liquid_data):
        """Assess contamination prevention techniques"""
        # Look for pauses that indicate tip changes or cleaning
        if liquid_data.empty or 'dominant_hand_velocity' not in liquid_data:
            return 0.5
        
        velocity = liquid_data['dominant_hand_velocity'].fillna(0)
        # Identify low-activity periods (potential tip changes)
        low_activity_threshold = velocity.mean() * 0.1
        low_activity_periods = (velocity < low_activity_threshold).sum()
        
        # More low-activity periods suggest better contamination prevention
        total_periods = len(velocity)
        contamination_prevention_score = min(1.0, low_activity_periods / (total_periods * 0.1))
        
        return contamination_prevention_score
    
    def _assess_reverse_pipetting_technique(self, df):
        """Assess reverse pipetting technique for glycerol handling"""
        # Identify glycerol handling period (would be more sophisticated in practice)
        glycerol_start = len(df) * 0.5  # Assume glycerol is in second half
        glycerol_end = len(df) * 0.75
        glycerol_data = df.iloc[int(glycerol_start):int(glycerol_end)]
        
        return self._detect_reverse_pipetting_pattern(glycerol_data)
    
    def _assess_pre_wetting_technique(self, df):
        """Assess pre-wetting technique for ethanol handling"""
        # Identify ethanol handling period (would be more sophisticated in practice)
        ethanol_start = len(df) * 0.75  # Assume ethanol is in last quarter
        ethanol_end = len(df)
        ethanol_data = df.iloc[int(ethanol_start):int(ethanol_end)]
        
        return self._detect_pre_wetting_pattern(ethanol_data)
    
    def _assess_tip_change_technique(self, df):
        """Assess tip change technique"""
        if df.empty or 'dominant_hand_velocity' not in df.columns:
            return 0.5
        
        velocity = df['dominant_hand_velocity'].fillna(0)
        # Look for patterns of low activity (potential tip changes)
        low_activity_threshold = velocity.quantile(0.1)
        low_activity_periods = (velocity < low_activity_threshold)
        
        # Identify continuous periods of low activity
        low_periods = []
        current_period = 0
        for is_low in low_activity_periods:
            if is_low:
                current_period += 1
            else:
                if current_period > 0:
                    low_periods.append(current_period)
                current_period = 0
        
        # More distinct low periods suggest better tip change technique
        expected_tip_changes = 4  # Based on protocol
        tip_change_score = min(1.0, len(low_periods) / expected_tip_changes)
        return tip_change_score
    
    def _assess_contamination_prevention_overall(self, df):
        """Assess overall contamination prevention across all liquids"""
        if df.empty:
            return 0.5
        
        contamination_scores = []
        for liquid in self.protocol_config['liquids']:
            liquid_data = self._extract_liquid_specific_data(df, liquid)
            score = self._assess_contamination_prevention(liquid_data)
            contamination_scores.append(score)
        
        return np.mean(contamination_scores) if contamination_scores else 0.5
    
    def _calculate_timing_variability(self, df):
        """Calculate timing variability across operations"""
        if df.empty or len(df) < 4:
            return 10.0  # High variability if insufficient data
        
        # Divide into quarters to simulate cycles
        quarter_size = len(df) // 4
        quarter_durations = []
        
        for i in range(4):
            start_idx = i * quarter_size
            end_idx = (i + 1) * quarter_size if i < 3 else len(df)
            quarter_data = df.iloc[start_idx:end_idx]
            
            if not quarter_data.empty:
                duration = quarter_data['time_seconds'].iloc[-1] - quarter_data['time_seconds'].iloc[0]
                quarter_durations.append(duration)
        
        if len(quarter_durations) > 1:
            return np.std(quarter_durations)
        else:
            return 0.0
    
    def _calculate_spatial_variability(self, df):
        """Calculate spatial variability in hand positions"""
        if df.empty or 'dom_hand_x' not in df.columns or 'dom_hand_y' not in df.columns:
            return 50.0  # High variability if no position data
        
        # Calculate center of activity
        valid_positions = df.dropna(subset=['dom_hand_x', 'dom_hand_y'])
        if len(valid_positions) < 2:
            return 50.0
        
        center_x = valid_positions['dom_hand_x'].mean()
        center_y = valid_positions['dom_hand_y'].mean()
        
        # Calculate distances from center
        distances = np.sqrt((valid_positions['dom_hand_x'] - center_x)**2 + 
                           (valid_positions['dom_hand_y'] - center_y)**2)
        
        return distances.std()
    
    def _calculate_velocity_variability(self, df):
        """Calculate velocity variability"""
        if df.empty or 'dominant_hand_velocity' not in df.columns:
            return 25.0  # High variability if no velocity data
        
        velocity_data = df['dominant_hand_velocity'].dropna()
        if len(velocity_data) < 2:
            return 25.0
        
        return velocity_data.std()
        # Identify ethanol handling period (assume last quarter)
        ethanol_start = len(df) * 0.75
        ethanol_data = df.iloc[int(ethanol_start):]
        
        return self._detect_pre_wetting_pattern(ethanol_data)
    
    def _assess_tip_change_technique(self, df):
        """Assess tip changing technique"""
        # Look for periods of very low activity followed by resumption
        if df.empty or 'dominant_hand_velocity' not in df:
            return 0.5
        
        velocity = df['dominant_hand_velocity'].fillna(0)
        
        # Find periods of minimal movement (tip changes)
        minimal_movement_threshold = velocity.mean() * 0.05
        minimal_periods = velocity < minimal_movement_threshold
        
        # Find transitions from minimal to active movement
        transitions = np.diff(minimal_periods.astype(int))
        tip_change_events = np.sum(transitions == -1)  # Transitions from minimal to active
        
        expected_tip_changes = self.protocol_config['tip_changes']
        tip_change_score = min(1.0, tip_change_events / expected_tip_changes)
        
        return tip_change_score
    
    def _assess_contamination_prevention_overall(self, df):
        """Assess overall contamination prevention throughout protocol"""
        if df.empty:
            return 0.5
        
        # Look for appropriate pauses between different liquids
        expected_liquid_boundaries = [0.25, 0.5, 0.75]  # Approximate liquid transition points
        contamination_scores = []
        
        for boundary in expected_liquid_boundaries:
            boundary_idx = int(len(df) * boundary)
            boundary_window = df.iloc[max(0, boundary_idx-10):min(len(df), boundary_idx+10)]
            
            if not boundary_window.empty and 'dominant_hand_velocity' in boundary_window:
                # Check for reduced activity at boundary (indicates proper procedure)
                boundary_activity = boundary_window['dominant_hand_velocity'].fillna(0).mean()
                overall_activity = df['dominant_hand_velocity'].fillna(0).mean()
                
                if boundary_activity < overall_activity * 0.5:
                    contamination_scores.append(1.0)
                else:
                    contamination_scores.append(0.3)
        
        return np.mean(contamination_scores) if contamination_scores else 0.5
    
    def _calculate_timing_variability(self, df):
        """Calculate timing variability across cycles"""
        if df.empty or len(df) < 4:
            return 0.5
        
        # Divide into expected cycles and measure timing
        cycles_per_liquid = len(df) // 4  # 4 liquids
        cycle_durations = []
        
        for i in range(4):
            cycle_start = i * cycles_per_liquid
            cycle_end = (i + 1) * cycles_per_liquid
            cycle_data = df.iloc[cycle_start:cycle_end]
            
            if not cycle_data.empty:
                duration = cycle_data['time_seconds'].iloc[-1] - cycle_data['time_seconds'].iloc[0]
                cycle_durations.append(duration)
        
        if len(cycle_durations) < 2:
            return 0.5
        
        cv = np.std(cycle_durations) / np.mean(cycle_durations)
        return cv
    
    def _calculate_spatial_variability(self, df):
        """Calculate spatial movement variability across cycles"""
        if df.empty:
            return 0.5
        
        # Analyze dominant hand position variability
        if 'dominant_hand' not in df.columns:
            return 0.5
        
        # Extract x, y coordinates where available
        positions = []
        for _, row in df.iterrows():
            if pd.notna(row['dominant_hand']) and row['dominant_hand']:
                try:
                    # Assuming dominant_hand contains coordinate information
                    if isinstance(row['dominant_hand'], dict) and 'center' in row['dominant_hand']:
                        positions.append(row['dominant_hand']['center'])
                except:
                    continue
        
        if len(positions) < 10:
            return 0.5
        
        # Calculate coefficient of variation in positions
        x_coords = [pos[0] for pos in positions if len(pos) >= 2]
        y_coords = [pos[1] for pos in positions if len(pos) >= 2]
        
        if not x_coords or not y_coords:
            return 0.5
        
        x_cv = np.std(x_coords) / (np.mean(x_coords) + 0.001)
        y_cv = np.std(y_coords) / (np.mean(y_coords) + 0.001)
        
        spatial_variability = (x_cv + y_cv) / 2.0
        return spatial_variability
    
    def _calculate_velocity_variability(self, df):
        """Calculate velocity pattern variability across cycles"""
        if df.empty or 'dominant_hand_velocity' not in df:
            return 0.5
        
        velocity = df['dominant_hand_velocity'].fillna(0)
        
        # Divide into cycles and analyze velocity patterns
        cycle_length = len(velocity) // 4
        cycle_velocity_stats = []
        
        for i in range(4):
            start_idx = i * cycle_length
            end_idx = (i + 1) * cycle_length
            cycle_vel = velocity.iloc[start_idx:end_idx]
            
            if len(cycle_vel) > 0:
                cycle_velocity_stats.append({
                    'mean': cycle_vel.mean(),
                    'std': cycle_vel.std(),
                    'max': cycle_vel.max()
                })
        
        if len(cycle_velocity_stats) < 2:
            return 0.5
        
        # Calculate variability in velocity statistics across cycles
        means = [stat['mean'] for stat in cycle_velocity_stats]
        stds = [stat['std'] for stat in cycle_velocity_stats]
        
        mean_cv = np.std(means) / (np.mean(means) + 0.001)
        std_cv = np.std(stds) / (np.mean(stds) + 0.001)
        
        velocity_variability = (mean_cv + std_cv) / 2.0
        return velocity_variability

    def assess_operator_type(self, metrics):
        """Assess if operator is freshly trained student, experienced worker, or automated"""
        reproducibility = metrics.get('inter_cycle_variability', {}).get('overall_reproducibility_score', 0.5)
        timing_consistency = metrics.get('timing_metrics', {}).get('cycle_consistency_score', 0.5) if 'timing_metrics' in metrics else 0.5
        technique_adherence = np.mean([
            metrics.get('technique_adherence', {}).get('reverse_pipetting_glycerol', 0.5),
            metrics.get('technique_adherence', {}).get('pre_wetting_ethanol', 0.5)
        ])
        
        # Decision logic based on experimental expectations
        if reproducibility > 0.9 and timing_consistency > 0.9:
            return "automated_liquid_handler"
        elif technique_adherence > 0.8 and timing_consistency > 0.7:
            return "experienced_lab_worker"
        elif technique_adherence > 0.6 and reproducibility > 0.5:
            return "freshly_trained_student"
        else:
            return "novice_learning"
    
    def calculate_accuracy_predictions(self, metrics):
        """Predict accuracy based on observed technique metrics"""
        predictions = {}
        
        for volume in self.protocol_config['volumes']:
            volume_key = f'{volume}uL'
            consistency = metrics.get('volume_consistency', {}).get(volume_key, {})
            
            # Predict CV% based on movement consistency
            velocity_cv = consistency.get('velocity_consistency_cv', 20.0)
            duration_cv = consistency.get('duration_consistency_cv', 20.0)
            
            # Empirical model for CV prediction (would be calibrated with actual data)
            predicted_cv = (velocity_cv * 0.6 + duration_cv * 0.4) / 5.0  # Scale down
            
            # Accuracy target from protocol config
            target = self.protocol_config['accuracy_targets'][volume]
            target_cv = float(target['expected_cv'].replace('<', '').replace('%', ''))
            
            predictions[volume_key] = {
                'predicted_cv_percent': predicted_cv,
                'target_cv_percent': target_cv,
                'meets_target': predicted_cv <= target_cv,
                'accuracy_score': max(0.0, 1.0 - (predicted_cv / target_cv))
            }
        
        return predictions
        technique_adherence = np.mean([
            metrics.get('technique_adherence', {}).get('reverse_pipetting_glycerol', 0.5),
            metrics.get('technique_adherence', {}).get('pre_wetting_ethanol', 0.5)
        ])
        
        # Decision logic based on experimental expectations
        if reproducibility > 0.9 and timing_variability < 0.1:
            return "automated_liquid_handler"
        elif technique_adherence > 0.7 and reproducibility > 0.6:
            return "experienced_lab_worker"
        elif technique_adherence > 0.5 and reproducibility > 0.4:
            return "freshly_trained_student"
        else:
            return "novice_learning"
    
    def calculate_accuracy_predictions(self, metrics):
        """Predict accuracy based on observed technique metrics"""
        predictions = {}
        
        for volume in self.protocol_config['volumes']:
            volume_key = f'{volume}uL'
            volume_data = metrics.get('volume_consistency', {}).get(volume_key, {})
            
            # Base prediction on movement consistency
            duration_cv = volume_data.get('duration_consistency_cv', 20.0)
            velocity_cv = volume_data.get('velocity_consistency_cv', 20.0)
            
            # Empirical model for CV prediction (would be calibrated with actual data)
            predicted_cv = np.mean([duration_cv, velocity_cv]) * 0.5  # Scale down
            
            # Accuracy target from protocol config
            target = self.protocol_config['accuracy_targets'][volume]
            target_cv_str = target['expected_cv'].replace('<', '').replace('%', '')
            target_cv = float(target_cv_str)
            
            predictions[volume_key] = {
                'predicted_cv_percent': predicted_cv,
                'target_cv_percent': target_cv,
                'meets_target': predicted_cv <= target_cv,
                'accuracy_score': max(0.0, 1.0 - (predicted_cv / target_cv)) if target_cv > 0 else 0.5
            }
        
        return predictions

def main():
    parser = argparse.ArgumentParser(description='Protocol-Specific Pipetting Video Analysis')
    parser.add_argument('video_path', help='Path to the video file to analyze')
    parser.add_argument('--output-dir', '-o', default='protocol_analysis_results', 
                       help='Output directory for results')
    parser.add_argument('--max-frames', '-m', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--skip-frames', '-s', type=int, default=30,
                       help='Process every Nth frame (default: 30)')
    parser.add_argument('--no-gloves', action='store_true',
                       help='Disable glove detection mode')
    parser.add_argument('--hue-offset', type=int, default=90,
                       help='HSV hue offset for glove detection (0-179)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ProtocolPipettingAnalyzer(
        args.video_path,
        args.output_dir,
        glove_mode=not args.no_gloves,
        hue_offset=args.hue_offset
    )
    
    # Run analysis
    try:
        metrics = analyzer.run_analysis(
            max_frames=args.max_frames,
            skip_frames=args.skip_frames
        )
        print("\n✅ Protocol analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

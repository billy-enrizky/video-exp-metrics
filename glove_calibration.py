#!/usr/bin/env python3
"""
Glove Detection Calibration Tool
Helps find the optimal hue offset for detecting hands in gloves
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
from pathlib import Path

class GloveCalibrationTool:
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.mp_hands = mp.solutions.hands
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        # Current settings
        self.current_hue_offset = 90
        self.current_frame_idx = 0
        self.frames = []
        
        # Load sample frames
        self.load_sample_frames()
        
    def load_sample_frames(self, num_samples=10):
        """Load sample frames from the video"""
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly throughout the video
        frame_indices = np.linspace(0, frame_count-1, num_samples, dtype=int)
        
        print(f"Loading {num_samples} sample frames...")
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize for faster processing
                height, width = frame.shape[:2]
                if width > 1280:
                    scale = 1280 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                self.frames.append(frame)
        
        cap.release()
        print(f"Loaded {len(self.frames)} frames")
    
    def apply_hue_offset(self, image, hue_offset):
        """Apply HSV hue offset to improve glove detection"""
        if hue_offset == 0:
            return image
        
        # Convert to HSV
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        
        # Apply hue offset
        h_shifted = cv2.add(h, hue_offset)
        
        # Recombine and convert back
        img_hsv_shifted = cv2.merge([h_shifted, s, v])
        return cv2.cvtColor(img_hsv_shifted, cv2.COLOR_HSV2BGR)
    
    def detect_hands_with_settings(self, frame, hue_offset, use_channel_swap=False):
        """Detect hands with current settings"""
        if use_channel_swap:
            # Method 1: Simple BGR->RGB channel swap
            rgb_frame = frame  # Use BGR directly as RGB
        elif hue_offset > 0:
            # Method 2: HSV hue offset
            processed_frame = self.apply_hue_offset(frame, hue_offset)
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        else:
            # Standard processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(rgb_frame)
        
        # Draw results on original frame
        output_frame = frame.copy()
        hands_detected = 0
        
        if results.multi_hand_landmarks:
            hands_detected = len(results.multi_hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    output_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style()
                )
        
        return output_frame, hands_detected
    
    def on_hue_trackbar(self, val):
        """Trackbar callback for hue offset"""
        self.current_hue_offset = val
        self.update_display()
    
    def on_frame_trackbar(self, val):
        """Trackbar callback for frame selection"""
        self.current_frame_idx = val
        self.update_display()
    
    def update_display(self):
        """Update the display with current settings"""
        if not self.frames:
            return
        
        frame = self.frames[self.current_frame_idx]
        
        # Test different methods
        # Original
        original = frame.copy()
        cv2.putText(original, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Channel swap method
        swap_frame, swap_hands = self.detect_hands_with_settings(frame, 0, use_channel_swap=True)
        cv2.putText(swap_frame, f"Channel Swap: {swap_hands} hands", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # HSV offset method
        hsv_frame, hsv_hands = self.detect_hands_with_settings(frame, self.current_hue_offset, use_channel_swap=False)
        cv2.putText(hsv_frame, f"HSV Offset {self.current_hue_offset}: {hsv_hands} hands", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Standard method (for comparison)
        std_frame, std_hands = self.detect_hands_with_settings(frame, 0, use_channel_swap=False)
        cv2.putText(std_frame, f"Standard: {std_hands} hands", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Combine into grid
        top_row = np.hstack([original, std_frame])
        bottom_row = np.hstack([swap_frame, hsv_frame])
        combined = np.vstack([top_row, bottom_row])
        
        # Resize if too large
        height, width = combined.shape[:2]
        if width > 1600:
            scale = 1600 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            combined = cv2.resize(combined, (new_width, new_height))
        
        cv2.imshow("Glove Detection Calibration", combined)
    
    def run_calibration(self):
        """Run the interactive calibration tool"""
        if not self.frames:
            print("No frames loaded!")
            return
        
        window_name = "Glove Detection Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Create trackbars
        cv2.createTrackbar("Hue Offset", window_name, self.current_hue_offset, 179, self.on_hue_trackbar)
        cv2.createTrackbar("Frame", window_name, self.current_frame_idx, len(self.frames)-1, self.on_frame_trackbar)
        
        print("\n🧤 Glove Detection Calibration Tool")
        print("=" * 50)
        print("Instructions:")
        print("- Use 'Hue Offset' slider to adjust HSV hue offset (0-179)")
        print("- Use 'Frame' slider to switch between sample frames")
        print("- Compare detection results in the four panels:")
        print("  • Top-left: Original frame")
        print("  • Top-right: Standard detection")
        print("  • Bottom-left: Channel swap method")
        print("  • Bottom-right: HSV offset method")
        print("- Press 'q' to quit and save best settings")
        print("- Press 's' to save current frame as test image")
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                frame = self.frames[self.current_frame_idx]
                save_path = f"test_frame_{self.current_frame_idx}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"Saved frame to {save_path}")
            elif key == ord('h'):
                # Show help
                print("\nHelp:")
                print("q - Quit")
                print("s - Save current frame")
                print("h - Show this help")
        
        cv2.destroyAllWindows()
        
        # Print recommendations
        print(f"\n📊 CALIBRATION RESULTS")
        print("=" * 50)
        print(f"Recommended settings for your video:")
        print(f"  HSV Hue Offset: {self.current_hue_offset}")
        print(f"  Frame tested: {self.current_frame_idx + 1}/{len(self.frames)}")
        print(f"\nTo use these settings with the analyzer:")
        print(f"  python real_video_analyzer.py {self.video_path.name} --hue-offset {self.current_hue_offset}")
        print(f"  python real_video_analyzer.py {self.video_path.name} --hue-offset 0  # for channel swap method")

def main():
    parser = argparse.ArgumentParser(description='Glove Detection Calibration Tool')
    parser.add_argument('video_path', help='Path to the video file to calibrate')
    args = parser.parse_args()
    
    try:
        tool = GloveCalibrationTool(args.video_path)
        tool.run_calibration()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

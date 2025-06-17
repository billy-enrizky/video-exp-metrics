#!/usr/bin/env python3
"""
Reproducibility Experiment Analysis
Specifically designed for the liquid handling reproducibility study comparing:
- Freshly trained students
- Experienced lab workers  
- Automated liquid handlers

Protocol: 4 volumes (50, 100, 150, 200 μL) × 4 liquids (DYE ddH2O, DYE-FREE ddH2O, GLYCEROL, ETHANOL)
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
from protocol_analyzer import ProtocolPipettingAnalyzer
from protocol_metrics_plotter_fixed import ProtocolMetricsPlotter
import warnings
warnings.filterwarnings('ignore')

class ReproducibilityExperimentAnalyzer:
    def __init__(self, output_dir="reproducibility_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Experiment-specific configuration
        self.experiment_config = {
            'study_design': 'reproducibility_comparison',
            'protocol': {
                'volumes_ul': [50, 100, 150, 200],
                'liquids': ['DYE_ddH2O', 'DYE-FREE_ddH2O', 'GLYCEROL', 'ETHANOL'],
                'total_dispensing_events': 16,
                'special_techniques': {
                    'GLYCEROL': 'reverse_pipetting',
                    'ETHANOL': 'pre_wetting'
                }
            },
            'operator_categories': [
                'freshly_trained_student',
                'experienced_lab_worker', 
                'automated_liquid_handler'
            ],
            'primary_metrics': [
                'accuracy_coefficient_of_variation',
                'inter_operator_consistency',
                'technique_adherence_score',
                'temporal_reproducibility'
            ],
            'accuracy_targets': {
                50: {'cv_target': 5.0, 'tolerance': 2.0},   # 5% CV, ±2% tolerance
                100: {'cv_target': 3.0, 'tolerance': 1.5},  # 3% CV, ±1.5% tolerance
                150: {'cv_target': 2.5, 'tolerance': 1.3},  # 2.5% CV, ±1.3% tolerance
                200: {'cv_target': 2.0, 'tolerance': 1.0}   # 2% CV, ±1% tolerance
            }
        }
        
        self.analysis_results = {}
        
    def analyze_operator_video(self, video_path, operator_type, operator_id=None):
        """Analyze a single operator's video for reproducibility metrics"""
        print(f"\n🔬 Analyzing {operator_type} - Video: {Path(video_path).name}")
        print("=" * 80)
        
        # Initialize protocol analyzer
        analyzer = ProtocolPipettingAnalyzer(
            video_path=video_path,
            output_dir=self.output_dir / f"{operator_type}_{operator_id or 'unknown'}",
            glove_mode=True,
            hue_offset=135  # Optimal value for glove detection
        )
        
        # Run analysis with appropriate sampling for reproducibility focus
        max_frames = 5000  # Process more frames for better accuracy
        skip_frames = 15    # Higher resolution sampling
        
        results = analyzer.run_analysis(max_frames=max_frames, skip_frames=skip_frames)
        
        # Extract reproducibility-specific metrics
        reproducibility_score = results['protocol_analysis']['reproducibility_analysis']['overall_reproducibility_score']
        accuracy_predictions = results['protocol_analysis']['reproducibility_analysis'].get('accuracy_predictions', {})
        
        # Store results
        analysis_key = f"{operator_type}_{operator_id or datetime.now().strftime('%H%M%S')}"
        self.analysis_results[analysis_key] = {
            'operator_type': operator_type,
            'operator_id': operator_id,
            'video_path': str(video_path),
            'reproducibility_score': reproducibility_score,
            'accuracy_predictions': accuracy_predictions,
            'full_analysis': results,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Generate reproducibility-focused report
        self._generate_reproducibility_report(analysis_key)
        
        # Generate protocol metrics plots
        self._generate_protocol_plots(analysis_key)
        
        return results
    
    def _generate_protocol_plots(self, analysis_key):
        """Generate comprehensive protocol metrics plots"""
        data = self.analysis_results[analysis_key]
        
        # Find the protocol analysis data file
        operator_dir = self.output_dir / f"{data['operator_type']}_{data['operator_id'] or 'unknown'}"
        data_files = list(operator_dir.glob("protocol_analysis_data_*.json"))
        
        if data_files:
            data_path = data_files[0]
            plot_output_dir = operator_dir / "protocol_plots"
            
            try:
                print(f"\n🎨 Generating protocol metrics plots...")
                plotter = ProtocolMetricsPlotter(data_path, plot_output_dir)
                plots_created = plotter.generate_all_plots()
                
                print(f"✅ Generated {len(plots_created)} protocol plots")
                return plots_created
                
            except Exception as e:
                print(f"❌ Failed to generate plots: {e}")
                import traceback
                traceback.print_exc()
                return []
        else:
            print(f"⚠️ No protocol analysis data found for plotting")
            return []
    
    def _generate_reproducibility_report(self, analysis_key):
        """Generate a reproducibility-focused analysis report"""
        data = self.analysis_results[analysis_key]
        operator_type = data['operator_type']
        
        report_path = self.output_dir / f"reproducibility_report_{analysis_key}.txt"
        
        with open(report_path, 'w') as f:
            f.write("LIQUID HANDLING REPRODUCIBILITY EXPERIMENT ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Operator Type: {operator_type.replace('_', ' ').title()}\n")
            f.write(f"Video: {Path(data['video_path']).name}\n")
            f.write(f"Analysis Date: {data['analysis_timestamp'][:19].replace('T', ' ')}\n\n")
            
            # Protocol execution summary
            protocol = data['full_analysis']['protocol_analysis']
            f.write("PROTOCOL EXECUTION SUMMARY:\n")
            f.write(f"  Total Dispensing Events: {protocol['protocol_events']['total_dispensing_events']}/16 expected\n")
            f.write(f"  Protocol Completion: {protocol['protocol_events']['protocol_completion_percentage']:.1f}%\n")
            f.write(f"  Execution Time: {protocol['timing_metrics']['total_time_seconds']/60:.1f} minutes\n")
            f.write(f"  Time Consistency Score: {protocol['timing_metrics']['cycle_consistency_score']:.3f}\n\n")
            
            # Reproducibility analysis
            f.write("REPRODUCIBILITY METRICS:\n")
            f.write(f"  Overall Reproducibility Score: {data['reproducibility_score']:.3f}/1.0\n")
            
            if data['accuracy_predictions']:
                f.write("\nACCURACY PREDICTIONS BY VOLUME:\n")
                for volume_key, prediction in data['accuracy_predictions'].items():
                    volume = int(volume_key.replace('uL', ''))
                    target = self.experiment_config['accuracy_targets'][volume]
                    
                    f.write(f"  {volume} μL:\n")
                    f.write(f"    Predicted CV: {prediction['predicted_cv_percent']:.2f}%\n")
                    f.write(f"    Target CV: <{target['cv_target']:.1f}%\n")
                    f.write(f"    Meets Target: {'✓ YES' if prediction['meets_target'] else '✗ NO'}\n")
                    f.write(f"    Accuracy Score: {prediction['accuracy_score']:.3f}/1.0\n\n")
                    
                    f.write(f"  {volume} μL:\n")
                    f.write(f"    Predicted CV: {prediction['predicted_cv_percent']:.2f}%\n")
                    f.write(f"    Target CV: <{target['cv_target']:.1f}%\n")
                    f.write(f"    Meets Target: {'✓ YES' if prediction['meets_target'] else '✗ NO'}\n")
                    f.write(f"    Accuracy Score: {prediction['accuracy_score']:.3f}/1.0\n\n")
            
            # Technique adherence
            repro_metrics = protocol['reproducibility_analysis'].get('reproducibility_metrics', {})
            if 'technique_adherence' in repro_metrics:
                tech = repro_metrics['technique_adherence']
                f.write("TECHNIQUE ADHERENCE:\n")
                f.write(f"  Reverse Pipetting (Glycerol): {tech.get('reverse_pipetting_glycerol', 0):.2f}/1.0\n")
                f.write(f"  Pre-wetting (Ethanol): {tech.get('pre_wetting_ethanol', 0):.2f}/1.0\n")
                f.write(f"  Contamination Prevention: {tech.get('cross_contamination_prevention', 0):.2f}/1.0\n\n")
            
            # Operator classification
            skill = protocol['skill_assessment']
            f.write("OPERATOR CLASSIFICATION:\n")
            f.write(f"  Classification: {skill['operator_type_classification']}\n")
            f.write(f"  Confidence: {skill['overall_skill_score']:.2f}/1.0\n")
            f.write(f"  Consistency Rating: {skill['consistency_rating']}\n")
            
        print(f"📋 Reproducibility report saved: {report_path}")
        return report_path
    
    def compare_operators(self, min_analyses=2):
        """Compare reproducibility metrics across different operator types"""
        if len(self.analysis_results) < min_analyses:
            print(f"Need at least {min_analyses} analyses for comparison. Current: {len(self.analysis_results)}")
            return
        
        print(f"\n📊 Comparing {len(self.analysis_results)} operator analyses...")
        
        # Organize results by operator type
        by_operator_type = {}
        for key, data in self.analysis_results.items():
            op_type = data['operator_type']
            if op_type not in by_operator_type:
                by_operator_type[op_type] = []
            by_operator_type[op_type].append(data)
        
        # Generate comparison plots
        self._create_reproducibility_comparison_plots(by_operator_type)
        
        # Generate comparison report
        self._generate_comparison_report(by_operator_type)
        
        return by_operator_type
    
    def _create_reproducibility_comparison_plots(self, by_operator_type):
        """Create visual comparisons of reproducibility metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Liquid Handling Reproducibility Experiment Results', fontsize=16, fontweight='bold')
        
        # 1. Reproducibility scores by operator type
        ax1 = axes[0, 0]
        operator_types = []
        repro_scores = []
        
        for op_type, analyses in by_operator_type.items():
            for analysis in analyses:
                operator_types.append(op_type.replace('_', ' ').title())
                repro_scores.append(analysis['reproducibility_score'])
        
        df_repro = pd.DataFrame({
            'Operator Type': operator_types,
            'Reproducibility Score': repro_scores
        })
        
        sns.boxplot(data=df_repro, x='Operator Type', y='Reproducibility Score', ax=ax1)
        ax1.set_title('Reproducibility Scores by Operator Type')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Accuracy predictions by volume
        ax2 = axes[0, 1]
        volume_data = []
        
        for op_type, analyses in by_operator_type.items():
            for analysis in analyses:
                predictions = analysis.get('accuracy_predictions', {})
                for vol_key, pred in predictions.items():
                    volume = int(vol_key.replace('uL', ''))
                    volume_data.append({
                        'Operator Type': op_type.replace('_', ' ').title(),
                        'Volume (μL)': volume,
                        'Predicted CV (%)': pred['predicted_cv_percent'],
                        'Meets Target': pred['meets_target']
                    })
        
        if volume_data:
            df_vol = pd.DataFrame(volume_data)
            for volume in [50, 100, 150, 200]:
                target_cv = self.experiment_config['accuracy_targets'][volume]['cv_target']
                ax2.axhline(y=target_cv, color='red', linestyle='--', alpha=0.7, 
                           label=f'{volume}μL Target' if volume == 50 else "")
            
            sns.scatterplot(data=df_vol, x='Volume (μL)', y='Predicted CV (%)', 
                           hue='Operator Type', style='Meets Target', s=100, ax=ax2)
            ax2.set_title('Predicted CV% by Volume and Operator Type')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Timing consistency
        ax3 = axes[1, 0]
        timing_data = []
        
        for op_type, analyses in by_operator_type.items():
            for analysis in analyses:
                protocol = analysis['full_analysis']['protocol_analysis']
                timing_data.append({
                    'Operator Type': op_type.replace('_', ' ').title(),
                    'Cycle Consistency': protocol['timing_metrics']['cycle_consistency_score'],
                    'Time Efficiency': protocol['timing_metrics']['time_efficiency_score']
                })
        
        if timing_data:
            df_timing = pd.DataFrame(timing_data)
            sns.scatterplot(data=df_timing, x='Time Efficiency', y='Cycle Consistency',
                           hue='Operator Type', s=100, ax=ax3)
            ax3.set_title('Timing Performance: Consistency vs Efficiency')
            ax3.set_xlim(0, 2)
            ax3.set_ylim(0, 1)
        
        # 4. Technique adherence radar chart (simplified as bar chart)
        ax4 = axes[1, 1]
        technique_scores = []
        
        for op_type, analyses in by_operator_type.items():
            for analysis in analyses:
                repro_metrics = analysis['full_analysis']['protocol_analysis']['reproducibility_analysis'].get('reproducibility_metrics', {})
                tech = repro_metrics.get('technique_adherence', {})
                
                technique_scores.append({
                    'Operator Type': op_type.replace('_', ' ').title(),
                    'Reverse Pipetting': tech.get('reverse_pipetting_glycerol', 0),
                    'Pre-wetting': tech.get('pre_wetting_ethanol', 0),
                    'Contamination Prevention': tech.get('cross_contamination_prevention', 0)
                })
        
        if technique_scores:
            df_tech = pd.DataFrame(technique_scores)
            df_tech_melted = df_tech.melt(id_vars=['Operator Type'], 
                                         value_vars=['Reverse Pipetting', 'Pre-wetting', 'Contamination Prevention'],
                                         var_name='Technique', value_name='Score')
            
            sns.barplot(data=df_tech_melted, x='Technique', y='Score', hue='Operator Type', ax=ax4)
            ax4.set_title('Technique Adherence Scores')
            ax4.set_ylim(0, 1)
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'reproducibility_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Comparison plots saved: {plot_path}")
    
    def _generate_comparison_report(self, by_operator_type):
        """Generate comprehensive comparison report"""
        report_path = self.output_dir / 'reproducibility_comparison_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("LIQUID HANDLING REPRODUCIBILITY EXPERIMENT - COMPARATIVE ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Operators Analyzed: {len(self.analysis_results)}\n")
            f.write(f"Operator Types: {', '.join(by_operator_type.keys())}\n\n")
            
            # Summary statistics by operator type
            f.write("REPRODUCIBILITY SUMMARY BY OPERATOR TYPE:\n")
            f.write("-" * 50 + "\n")
            
            for op_type, analyses in by_operator_type.items():
                f.write(f"\n{op_type.replace('_', ' ').title()} (n={len(analyses)}):\n")
                
                # Calculate statistics
                repro_scores = [a['reproducibility_score'] for a in analyses]
                f.write(f"  Reproducibility Score: {np.mean(repro_scores):.3f} ± {np.std(repro_scores):.3f}\n")
                
                # Accuracy predictions summary
                all_predictions = {}
                for analysis in analyses:
                    for vol_key, pred in analysis.get('accuracy_predictions', {}).items():
                        if vol_key not in all_predictions:
                            all_predictions[vol_key] = []
                        all_predictions[vol_key].append(pred['predicted_cv_percent'])
                
                f.write("  Predicted CV% by Volume:\n")
                for vol_key, cvs in all_predictions.items():
                    volume = int(vol_key.replace('uL', ''))
                    target = self.experiment_config['accuracy_targets'][volume]['cv_target']
                    f.write(f"    {volume}μL: {np.mean(cvs):.2f}% ± {np.std(cvs):.2f}% (Target: <{target}%)\n")
                
            # Recommendations
            f.write(f"\nRECOMMendations:\n")
            f.write("-" * 20 + "\n")
            
            # Find best performing operator type
            avg_scores = {}
            for op_type, analyses in by_operator_type.items():
                avg_scores[op_type] = np.mean([a['reproducibility_score'] for a in analyses])
            
            best_type = max(avg_scores, key=avg_scores.get)
            f.write(f"• Best overall reproducibility: {best_type.replace('_', ' ').title()}\n")
            
            # Volume-specific recommendations
            for volume in [50, 100, 150, 200]:
                target_cv = self.experiment_config['accuracy_targets'][volume]['cv_target']
                f.write(f"• {volume}μL target CV <{target_cv}%: ")
                
                meeting_target = []
                for op_type, analyses in by_operator_type.items():
                    type_predictions = []
                    for analysis in analyses:
                        pred = analysis.get('accuracy_predictions', {}).get(f'{volume}uL', {})
                        if pred.get('meets_target', False):
                            type_predictions.append(True)
                    
                    if type_predictions:
                        meeting_target.append(op_type)
                
                if meeting_target:
                    f.write(f"{'✓ ' + ', '.join([t.replace('_', ' ').title() for t in meeting_target])}\n")
                else:
                    f.write("⚠ None consistently meet target\n")
        
        print(f"📋 Comparison report saved: {report_path}")
        return report_path

def main():
    parser = argparse.ArgumentParser(description='Reproducibility Experiment Analysis')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--operator-type', required=True, 
                       choices=['freshly_trained_student', 'experienced_lab_worker', 'automated_liquid_handler'],
                       help='Type of operator in the video')
    parser.add_argument('--operator-id', help='Unique identifier for this operator')
    parser.add_argument('--output-dir', default='reproducibility_results', 
                       help='Output directory for results')
    parser.add_argument('--compare', action='store_true',
                       help='Generate comparison analysis (requires multiple previous analyses)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    experiment = ReproducibilityExperimentAnalyzer(output_dir=args.output_dir)
    
    # Analyze the video
    results = experiment.analyze_operator_video(
        video_path=args.video_path,
        operator_type=args.operator_type,
        operator_id=args.operator_id
    )
    
    print(f"\n✅ Analysis complete for {args.operator_type}")
    print(f"Reproducibility Score: {results['protocol_analysis']['reproducibility_analysis']['overall_reproducibility_score']:.3f}/1.0")
    
    # Generate comparison if requested
    if args.compare:
        experiment.compare_operators()

if __name__ == "__main__":
    main()

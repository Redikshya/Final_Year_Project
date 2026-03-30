import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

class ModelEvaluator:
    """Model evaluation system for Face Recognition Attendance System."""
    
    def __init__(self, results_folder='model_evaluation_results'):
        self.results_folder = results_folder
        self.ensure_results_folder()
        
    def ensure_results_folder(self):
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
            print(f"[INFO] Created results folder: {self.results_folder}")
    
    def evaluate_model(self, model, X_train, y_train, model_name="Custom_KNN"):
        """Evaluate model and generate metrics."""
        print(f"[INFO] Evaluating {model_name} model...")
        
        # Get predictions on training data
        y_pred = model.predict(X_train)
        y_true = y_train
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred)
        
        # Generate visualizations
        self._generate_confusion_matrix(y_true, y_pred, model_name)
        self._generate_accuracy_metrics(metrics, model_name)
        self._save_detailed_report(metrics, y_true, y_pred, model_name)
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        total_samples = len(y_true)
        correct_predictions = np.sum(y_true == y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'confusion_matrix': cm
        }
    
    def _generate_confusion_matrix(self, y_true, y_pred, model_name):
        """Generate confusion matrix visualization."""
        try:
            classes = np.unique(np.concatenate([y_true, y_pred]))
            cm = confusion_matrix(y_true, y_pred, labels=classes)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=classes, yticklabels=classes)
            
            plt.title(f'Confusion Matrix - {model_name}\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            accuracy = accuracy_score(y_true, y_pred)
            plt.text(0.02, 0.98, f'Accuracy: {accuracy:.2%}', 
                    transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white'))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_confusion_matrix_{timestamp}.png"
            filepath = os.path.join(self.results_folder, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[SUCCESS] Confusion matrix saved: {filepath}")
            
        except Exception as e:
            print(f"[ERROR] Failed to generate confusion matrix: {e}")
    
    def _generate_accuracy_metrics(self, metrics, model_name):
        """Generate accuracy metrics visualization."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Model Performance - {model_name}\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            
            # Overall metrics
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
            
            bars = ax1.bar(metric_names, metric_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
            ax1.set_title('Overall Performance')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1)
            
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Sample distribution
            sample_data = [metrics['correct_predictions'], 
                          metrics['total_samples'] - metrics['correct_predictions']]
            sample_labels = ['Correct', 'Incorrect']
            colors = ['#2ECC71', '#E74C3C']
            
            ax2.pie(sample_data, labels=sample_labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Prediction Distribution')
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_accuracy_metrics_{timestamp}.png"
            filepath = os.path.join(self.results_folder, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[SUCCESS] Accuracy metrics saved: {filepath}")
            
        except Exception as e:
            print(f"[ERROR] Failed to generate accuracy metrics: {e}")
    
    def _save_detailed_report(self, metrics, y_true, y_pred, model_name):
        """Save detailed evaluation report."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_detailed_report_{timestamp}.txt"
            filepath = os.path.join(self.results_folder, filename)
            
            with open(filepath, 'w') as f:
                f.write(f"MODEL EVALUATION REPORT\n")
                f.write(f"========================\n\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Samples: {metrics['total_samples']}\n\n")
                
                f.write(f"PERFORMANCE METRICS\n")
                f.write(f"------------------\n")
                f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall: {metrics['recall']:.4f}\n")
                f.write(f"F1-Score: {metrics['f1']:.4f}\n\n")
                
                f.write(f"PREDICTIONS\n")
                f.write(f"-----------\n")
                f.write(f"Correct: {metrics['correct_predictions']}\n")
                f.write(f"Incorrect: {metrics['total_samples'] - metrics['correct_predictions']}\n")
                f.write(f"Success Rate: {metrics['correct_predictions']/metrics['total_samples']:.2%}\n\n")
                
                f.write(f"CONFUSION MATRIX\n")
                f.write(f"----------------\n")
                f.write(f"{metrics['confusion_matrix']}\n\n")
                
                f.write(f"CLASSIFICATION REPORT\n")
                f.write(f"--------------------\n")
                f.write(classification_report(y_true, y_pred))
            
            print(f"[SUCCESS] Detailed report saved: {filepath}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save detailed report: {e}")
    
    def cleanup_old_results(self, keep_latest=2):
        """Keep only the latest evaluation results."""
        try:
            all_files = [f for f in os.listdir(self.results_folder) if f.endswith(('.png', '.txt'))]
            
            if not all_files:
                print("[INFO] No evaluation files to clean up")
                return
            
            print(f"[INFO] Found {len(all_files)} total evaluation files")
            
            # Group files by type
            file_groups = {}
            for file in all_files:
                if '_confusion_matrix_' in file:
                    key = 'confusion_matrix'
                elif '_accuracy_metrics_' in file:
                    key = 'accuracy_metrics'
                elif '_detailed_report_' in file:
                    key = 'detailed_report'
                else:
                    continue
                
                if key not in file_groups:
                    file_groups[key] = []
                file_groups[key].append(file)
            
            # Keep only latest files
            for key, files in file_groups.items():
                print(f"[INFO] Processing {key}: {len(files)} files found")
                
                if len(files) > keep_latest:
                    # Extract timestamp properly from filename
                    def extract_timestamp(filename):
                        try:
                            # Filename format: Custom_KNN_Test_confusion_matrix_20251110_082439.png
                            # Split by underscore and remove extension
                            parts = filename.replace('.png', '').replace('.txt', '').split('_')
                            
                            # The timestamp is always the last 2 parts before extension
                            # Format: YYYYMMDD_HHMMSS
                            if len(parts) >= 2:
                                date_part = parts[-2]  # 20251110
                                time_part = parts[-1]  # 082439
                                
                                # Validate it's actually a timestamp
                                if len(date_part) == 8 and len(time_part) == 6:
                                    if date_part.isdigit() and time_part.isdigit():
                                        # Return as sortable string
                                        return f"{date_part}_{time_part}"
                            
                            # Fallback: return filename
                            print(f"[WARNING] Could not parse timestamp from: {filename}")
                            return filename
                            
                        except Exception as e:
                            print(f"[WARNING] Failed to parse timestamp from {filename}: {e}")
                            return filename
                    
                    # Sort by timestamp (newest first = reverse order)
                    files.sort(key=extract_timestamp, reverse=True)
                    
                    # Keep the first 'keep_latest' files (newest)
                    files_to_keep = files[:keep_latest]
                    files_to_delete = files[keep_latest:]
                    
                    print(f"[DEBUG] {key}: Keeping {len(files_to_keep)} newest file(s)")
                    print(f"[DEBUG] {key}: Deleting {len(files_to_delete)} oldest file(s)")
                    print(f"[DEBUG] {key}: Files to keep: {files_to_keep}")
                    print(f"[DEBUG] {key}: Files to delete: {files_to_delete}")
                    
                    # Delete old files
                    for old_file in files_to_delete:
                        filepath = os.path.join(self.results_folder, old_file)
                        try:
                            os.remove(filepath)
                            print(f"[INFO] Removed old file: {old_file}")
                        except Exception as e:
                            print(f"[ERROR] Failed to remove {old_file}: {e}")
                else:
                    print(f"[INFO] {key}: Only {len(files)} file(s), no cleanup needed")
            
            print(f"[SUCCESS] Cleanup completed. Kept latest {keep_latest} file(s) of each type.")
            
        except Exception as e:
            print(f"[ERROR] Failed to cleanup old results: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    evaluator = ModelEvaluator()
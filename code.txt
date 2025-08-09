import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch

def train_pomegranate_model():
    """Train a YOLO model for pomegranate disease detection"""
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = YOLO('yolov8n.pt').to(device)
    torch.cuda.empty_cache()

    # Basic training configuration
    train_params = {
        "data": "data.yaml",
        "epochs": 50,
        "imgsz": 640,
        "batch": 16,
        "name": "pomegranate_detection",
        "save": True,
        "patience": 10,
        "optimizer": "Adam",
        "augment": True,
    }

    # Train the model
    results = model.train(**train_params)
    
    # Evaluate and export
    metrics = model.val()
    model.export(format="onnx")
    
    return results, metrics

def create_comparison_framework():
    """Create a framework for performance comparison without specific numbers"""
    categories = {
        'All': ['Precision', 'Recall', 'F1-Score'],
        'Bud': ['Precision', 'Recall'],
        'Flower': ['F1-Score'],
        'Early-Fruit': ['Specificity'],
        'Mid-Growth': ['Precision'],
        'Mature': ['F1-score']
    }
    
    methods = [
        'Proposed Method',
        'YOLOv7',
        'Faster R-CNN',
        'EfficientDet',
        'SSD',
        'RetinaNet',
        'SVM',
        'Random Forest'
    ]
    
    # Create empty comparison table
    rows = []
    for category, metrics in categories.items():
        for metric in metrics:
            rows.append({'Category': category, 'Metric': metric})
    
    df = pd.DataFrame(rows)
    
    # Add empty columns for each method
    for method in methods:
        df[method] = ""
    df['Improvement'] = ""
    
    return df

def visualize_comparison(df):
    """Visualize the comparison framework"""
    plt.figure(figsize=(12, 6))
    
    # Create a simple table visualization
    cell_text = []
    for _, row in df.iterrows():
        cell_text.append([row['Category'], row['Metric']] + ['' for _ in range(len(df.columns)-2)])
    
    plt.table(cellText=cell_text,
              colLabels=df.columns,
              loc='center',
              cellLoc='center')
    
    plt.axis('off')
    plt.title("Performance Comparison Framework")
    plt.savefig('comparison_framework.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Train the model
    print("Starting model training...")
    train_results, val_metrics = train_pomegranate_model()
    
    # Create comparison framework
    print("\nCreating performance comparison framework...")
    comparison_df = create_comparison_framework()
    
    # Visualize and save
    visualize_comparison(comparison_df)
    comparison_df.to_csv('performance_framework.csv', index=False)
    
    print("\nTraining completed and framework created:")
    print("- Model saved")
    print("- Comparison framework saved to performance_framework.csv")
    print("- Framework visualization saved to comparison_framework.png")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(y_true, y_pred):
    """
    Calculates and prints classification metrics.
    
    Args:
        y_true: The true labels.
        y_pred: The predicted labels from the model.
        
    Returns:
        A dictionary containing the calculated metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print("--- Evaluation Metrics ---")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
        
    return metrics

if __name__ == '__main__':
    # This is an example of how you would use the evaluation function.
    # In a real scenario, y_true and y_pred would come from your model's output on a test set.
    
    print("--- Example Evaluation ---")
    # Dummy data for demonstration
    y_true_example = [0, 1, 0, 1, 1, 0, 0, 1]
    y_pred_example = [0, 1, 0, 0, 1, 1, 0, 1]
    
    evaluate_model(y_true_example, y_pred_example)


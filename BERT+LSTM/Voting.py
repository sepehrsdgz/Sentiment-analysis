import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from transformers import TFBertModel
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load models with custom objects for BERT
bert_model = load_model(r'C:\ITU\RFI-project\Model-training\BERT-4\BERT_V.1.4.h5', custom_objects={"TFBertModel": TFBertModel})
lstm_model = load_model(r'C:\ITU\RFI-project\Model-training\LSTM-3\LSTM-3.h5')

# Load test data
test_input_ids = pd.read_csv(r'C:\ITU\RFI-project\Model-training\BERT-4\test_input_ids.csv').values
test_attention_masks = pd.read_csv(r'C:\ITU\RFI-project\Model-training\BERT-4\test_attention_masks.csv').values
X_test_padded = pd.read_csv(r'C:\ITU\RFI-project\Model-training\LSTM-3\x_test_padded.csv').values
y_test = pd.read_csv(r'C:\ITU\RFI-project\Model-training\LSTM-3\y_test.csv').values

# Ensure test data shape consistency
print(f"BERT input shape: {test_input_ids.shape}, {test_attention_masks.shape}")
print(f"LSTM input shape: {X_test_padded.shape}")
print(f"y_test shape (one-hot): {y_test.shape}")

# Make predictions on the test set
bert_predictions = bert_model.predict([test_input_ids, test_attention_masks])
lstm_predictions = lstm_model.predict(X_test_padded)

# Check a few samples of raw model outputs
print("\nSample BERT predictions (probabilities):", bert_predictions[:5])
print("\nSample LSTM predictions (probabilities):", lstm_predictions[:5])

# Convert predictions to class labels
bert_pred_labels = np.argmax(bert_predictions, axis=1)
lstm_pred_labels = np.argmax(lstm_predictions, axis=1)

# Print sample labels for debugging
print("\nSample BERT class labels:", bert_pred_labels[:10])
print("\nSample LSTM class labels:", lstm_pred_labels[:10])

# Convert y_test to class labels for comparison
y_test_labels = np.argmax(y_test, axis=1)
print("\nTrue labels sample:", y_test_labels[:10])

# Evaluate individual model performances
print("\nBERT model performance:")
print(classification_report(y_test_labels, bert_pred_labels, target_names=['Negative', 'Neutral', 'Positive', 'Irrelevant']))

print("\nLSTM model performance:")
print(classification_report(y_test_labels, lstm_pred_labels, target_names=['Negative', 'Neutral', 'Positive', 'Irrelevant']))

# Ensemble Voting
ensemble_predictions = []
for bert_pred, lstm_pred in zip(bert_pred_labels, lstm_pred_labels):
    ensemble_predictions.append(bert_pred if bert_pred == lstm_pred else bert_pred)

# Evaluate ensemble performance
ensemble_report = classification_report(y_test_labels, ensemble_predictions, target_names=['Negative', 'Neutral', 'Positive', 'Irrelevant'])
print("\nClassification Report for Ensemble Voting:\n", ensemble_report)

# Generate confusion matrix
ensemble_cm = confusion_matrix(y_test_labels, ensemble_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(ensemble_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive', 'Irrelevant'],
            yticklabels=['Negative', 'Neutral', 'Positive', 'Irrelevant'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Ensemble Voting')
plt.show()


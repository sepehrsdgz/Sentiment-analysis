# Importing Libraries
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from transformers import TFBertModel, BertTokenizerFast
import tensorflow as tf
from sklearn.metrics import classification_report
import logging
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(filename='training_logs.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU (if you have multiple GPUs)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        logging.info(f"Using {len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s)")
    except RuntimeError as e:
        logging.error(f"Error setting GPU visibility: {e}")
else:
    logging.warning("No GPU found, TensorFlow will use the CPU.")

# Enable device placement logging to ensure operations are running on the GPU
tf.debugging.set_log_device_placement(True)

df_train = pd.read_csv(r'C:\ITU\RFI-proje\Twitter Sentiment Analysis DATA\twitter_training.csv')
df_val = pd.read_csv(r'C:\ITU\RFI-proje\Twitter Sentiment Analysis DATA\twitter_validation.csv')


# Function to remove \n and \r and lowercase
def remove_all_entities(text):
    if isinstance(text, str):
        text = text.replace('\r', '').replace('\n', ' ').lower()
        text = re.sub(r'[^\x00-\x7f]', r'', text)  # Remove non-UTF8 characters
        banned_list = string.punctuation + 'Ã' + '±' + 'ã' + '¼' + 'â' + '»' + '§'
        table = str.maketrans('', '', banned_list)
        text = text.translate(table)
        return text
    return text


# Function to remove URLs
def remove_urls(text):
    url_pattern = r'http\S+|www\S+'
    return re.sub(url_pattern, '', text) if isinstance(text, str) else text


# Function to remove special characters except "#"
def remove_non_word_chars(text):
    pattern = r'[^\w\s#]'
    return re.sub(pattern, '', text) if isinstance(text, str) else text


# Function to remove digits
def remove_digits(text):
    return re.sub(r'\d+', '', text) if isinstance(text, str) else text


# Function to remove emojis
def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text) if isinstance(text, str) else text


# Function to remove hashtags at the end of the sentence
def clean_hashtags(tweet):
    if isinstance(tweet, str):
        new_tweet = " ".join(
            word.strip() for word in re.split(r'#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet))
        new_tweet2 = " ".join(word.strip() for word in re.split(r'#|_', new_tweet))
        return new_tweet2
    return tweet


# Function to apply all cleaning steps
def clean_text(text):
    if isinstance(text, str):
        text = remove_all_entities(text)
        text = remove_urls(text)
        text = remove_non_word_chars(text)
        text = remove_digits(text)
        text = remove_emoji(text)
        text = clean_hashtags(text)
    return text


# Now apply this function to both training and validation datasets
df_train['cleaned_text'] = df_train['Tweet content'].apply(clean_text)
df_val['cleaned_text'] = df_val['Tweet content'].apply(clean_text)

# Check the length of the cleaned tweets
df_val['tweet_len'] = df_val['cleaned_text'].apply(lambda text: len(text.split()) if isinstance(text, str) else 0)
df_train['tweet_len'] = df_train['cleaned_text'].apply(lambda text: len(text.split()) if isinstance(text, str) else 0)

# Delete rows with tweet_len <= 2
df_train = df_train[df_train['tweet_len'] > 2]
df_val = df_val[df_val['tweet_len'] > 2]

# Map the sentiment labels to numbers
df_train['Sentiment'] = df_train['Sentiment'].map({'Neutral': 1, 'Negative': 0, 'Positive': 2, 'Irrelevant': 3})
df_val['Sentiment'] = df_val['Sentiment'].map({'Neutral': 1, 'Negative': 0, 'Positive': 2, 'Irrelevant': 3})

X_train, X_test, y_train, y_test = \
    train_test_split(df_train['cleaned_text'].values,
                     df_train['Sentiment'].values, test_size=0.3, random_state=42, stratify=df_train['Sentiment'])

# Print train/test split sample
print("\nSample X_train after split:")
print(X_train[:5])

# Use BertTokenizerFast for tokenization
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


# Define a function to tokenize and convert text to BERT format
def tokenize_data(texts, max_len=128):
    encodings = tokenizer(texts.tolist(), truncation=True, padding='max_length', max_length=max_len,
                          return_tensors='np')
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    return input_ids, attention_mask


# Print shapes to check

# Tokenize training, validation, and test data
train_input_ids, train_attention_masks = tokenize_data(X_train)
val_input_ids, val_attention_masks = tokenize_data(df_val['cleaned_text'])
test_input_ids, test_attention_masks = tokenize_data(X_test)

# Save tokenized data to CSV
np.savetxt("train_input_ids.csv", train_input_ids, delimiter=",")
np.savetxt("train_attention_masks.csv", train_attention_masks, delimiter=",")
np.savetxt("val_input_ids.csv", val_input_ids, delimiter=",")
np.savetxt("val_attention_masks.csv", val_attention_masks, delimiter=",")
np.savetxt("test_input_ids.csv", test_input_ids, delimiter=",")
np.savetxt("test_attention_masks.csv", test_attention_masks, delimiter=",")
np.savetxt("y_test.csv", y_test, delimiter=",")

# Print tokenization results
print("\nSample tokenized input IDs (train):")
print(train_input_ids[:2])
print("\nSample tokenized attention masks (train):")
print(train_attention_masks[:2])

print(f"Train Input IDs shape: {train_input_ids.shape}")
print(f"Train Attention Masks shape: {train_attention_masks.shape}")
print(f"Validation Input IDs shape: {val_input_ids.shape}")
print(f"Test Input IDs shape: {test_input_ids.shape}")

# Convert labels to one-hot encoded format
ohe = preprocessing.OneHotEncoder()
y_train = ohe.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
y_valid = ohe.transform(np.array(df_val['Sentiment']).reshape(-1, 1)).toarray()
y_test = ohe.transform(np.array(y_test).reshape(-1, 1)).toarray()

# Print one-hot encoded labels
print("\nSample one-hot encoded labels (train):")
print(y_train[:5])

bert_model = TFBertModel.from_pretrained('bert-base-uncased')


# Define a custom Keras model with BERT and classification layer
def create_model(bert_model, max_len=128):
    # Input layers for token IDs and attention masks
    input_ids = tf.keras.Input(shape=(max_len,), dtype='int32', name="input_ids")
    attention_masks = tf.keras.Input(shape=(max_len,), dtype='int32', name="attention_masks")

    # Pass inputs through BERT model
    embeddings = bert_model([input_ids, attention_masks])[1]  # Pooled output (last hidden state of the [CLS] token)

    # Add a dense layer for classification with 4 output classes (multi-class classification)
    output = tf.keras.layers.Dense(4, activation="softmax")(embeddings)

    # Create the final model
    model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Create the model
model = create_model(bert_model)

# Print model summary
model.summary()

# Define the batch size and number of epochs
BATCH_SIZE = 16
EPOCHS = 5

# Train the model
history = model.fit(
    [train_input_ids, train_attention_masks],  # Input data (tokenized inputs)
    y_train,  # One-hot encoded labels for training
    validation_data=([val_input_ids, val_attention_masks], y_valid),  # Validation data
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Save the model for future use
model.save('BERT_V.1.3.h5')
result_bert = model.predict([test_input_ids, test_attention_masks])
y_pred_bert = np.zeros_like(result_bert)
y_pred_bert[np.arange(len(y_pred_bert)), result_bert.argmax(1)] = 1
print('\tClassification Report for BERT:\n\n',
      classification_report(y_test, y_pred_bert, target_names=['Negative', 'Neutral', 'Positive', 'Irrelevant']))
#  Evaluate the model on the test set (optional)
test_loss, test_accuracy = model.evaluate([test_input_ids, test_attention_masks], y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Save the evaluation results
with open("evaluation_results.txt", "w") as f:
    f.write(f"Test Loss: {test_loss}\n")
    f.write(f"Test Accuracy: {test_accuracy}\n")

# Convert one-hot encoded labels back to the original label format
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred_bert, axis=1)

# Generate the confusion matrix
cm = confusion_matrix(y_test_labels, y_pred_labels)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive', 'Irrelevant'],
            yticklabels=['Negative', 'Neutral', 'Positive', 'Irrelevant'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for BERT-3 Sentiment Analysis')

# Save the plot as a PNG image
plt.savefig('confusion_matrix.png')

# Show the plot
plt.show()
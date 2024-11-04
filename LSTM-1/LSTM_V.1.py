
# For Data cleaning
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# For Tokenizing and padding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# For Training the LSTM Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

# For prediction results:
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



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

# Save cleaned data
df_train.to_csv('cleaned_train_data.csv', index=False)
df_val.to_csv('cleaned_val_data.csv', index=False)


X_train, X_test, y_train, y_test = \
    train_test_split(df_train['cleaned_text'].values,
                     df_train['Sentiment'].values, test_size=0.2, random_state=42, stratify=df_train['Sentiment'])

# Print train/test split sample
print("\nSample X_train after split:")
print(X_train[:5])

# Convert labels to one-hot encoded format
ohe = preprocessing.OneHotEncoder()
y_train = ohe.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
y_valid = ohe.transform(np.array(df_val['Sentiment']).reshape(-1, 1)).toarray()
y_test = ohe.transform(np.array(y_test).reshape(-1, 1)).toarray()



# Print one-hot encoded labels
print("\nSample one-hot encoded labels (train):")
print(y_train[:5])

# Tokenize and pad sequences
vocab_size = 10000
max_length = 50
oov_token = "<OOV>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(df_train['cleaned_text'].values)


# Convert texts to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_valid_seq = tokenizer.texts_to_sequences(df_val['cleaned_text'].values)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure equal length
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_valid_padded = pad_sequences(X_valid_seq, maxlen=max_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')


print("\nSample padded sequences (train):")
print(X_train_padded[:5])

# Model parameters
embedding_dim = 128  # Dimension of the embedding layer; can be adjusted
lstm_units = 128      # Number of units in the LSTM layer
dropout_rate = 0.5   # Dropout rate for regularization

# Initialize the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length, trainable=True),  # Using pretrained embeddings can be specified here
    Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)),
    LSTM(64, dropout=0.5, recurrent_dropout=0.5),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])


# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Convert padded sequences and labels to DataFrames
X_train_df = pd.DataFrame(X_train_padded)
y_train_df = pd.DataFrame(y_train)
X_valid_df = pd.DataFrame(X_valid_padded)
y_valid_df = pd.DataFrame(y_valid)
X_test_df = pd.DataFrame(X_test_padded)
y_test_df = pd.DataFrame(y_test)

# Save LSTM inputs to CSV files
X_train_df.to_csv('X_train_padded.csv', index=False)
y_train_df.to_csv('y_train.csv', index=False)
X_valid_df.to_csv('X_valid_padded.csv', index=False)
y_valid_df.to_csv('y_valid.csv', index=False)
X_test_df.to_csv('X_test_padded.csv', index=False)
y_test_df.to_csv('y_test.csv', index=False)


# Train the model
history = model.fit(
    X_train_padded, y_train,
    validation_data=(X_test_padded, y_test),
    epochs=15,
    batch_size=64,
    verbose=1
)

# Save the trained model
model.save('sentiment_analysis_lstm_model.h5')
print("Model saved as 'sentiment_analysis_lstm_model.h5'")


# After training, make predictions on the test set
y_pred = model.predict(X_test_padded)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predicted probabilities to class labels
y_test_classes = np.argmax(y_test, axis=1)  # Convert one-hot encoded true labels to class labels

# Generate the classification report
report = classification_report(y_test_classes, y_pred_classes, target_names=['Negative', 'Neutral', 'Positive', 'Irrelevant'])
print("Classification Report:\n", report)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive', 'Irrelevant'],
            yticklabels=['Negative', 'Neutral', 'Positive', 'Irrelevant'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for LSTM Sentiment Analysis')

# Save the plot as a PNG image
plt.savefig('confusion_matrix.png')

plt.show()

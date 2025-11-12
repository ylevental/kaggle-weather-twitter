import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Define target columns
sentiment_cols = ['s1', 's2', 's3', 's4', 's5']
when_cols = ['w1', 'w2', 'w3', 'w4']
kind_cols = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15']
all_target_cols = sentiment_cols + when_cols + kind_cols

print(f"Training samples: {len(train)}")
print(f"Test samples: {len(test)}")

# Text preprocessing
def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    # Keep important punctuation for sentiment
    text = re.sub(r'http\S+|www\.\S+', 'URL', text)
    text = re.sub(r'@\w+', 'MENTION', text)
    text = re.sub(r'#(\w+)', r'\1', text)  # Remove # but keep word
    return text

print("Preprocessing text...")
train['tweet_clean'] = train['tweet'].apply(preprocess_text)
test['tweet_clean'] = test['tweet'].apply(preprocess_text)

# Fill missing values
train['state'] = train['state'].fillna('unknown')
test['state'] = test['state'].fillna('unknown')
train['location'] = train['location'].fillna('unknown')
test['location'] = test['location'].fillna('unknown')

# Feature engineering
print("Creating features...")

# 1. TF-IDF features (more features)
tfidf = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)
X_train_tfidf = tfidf.fit_transform(train['tweet_clean'])
X_test_tfidf = tfidf.transform(test['tweet_clean'])

# 2. Character n-grams for style
char_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 4),
    max_features=500,
    min_df=3
)
X_train_char = char_vectorizer.fit_transform(train['tweet_clean'])
X_test_char = char_vectorizer.transform(test['tweet_clean'])

# 3. Text statistics features
def extract_text_features(df):
    features = pd.DataFrame()
    features['tweet_length'] = df['tweet_clean'].str.len()
    features['word_count'] = df['tweet_clean'].str.split().str.len()
    features['avg_word_length'] = features['tweet_length'] / (features['word_count'] + 1)
    features['exclamation_count'] = df['tweet'].str.count('!')
    features['question_count'] = df['tweet'].str.count('\?')
    features['caps_count'] = df['tweet'].str.count('[A-Z]')
    features['has_url'] = df['tweet_clean'].str.contains('URL').astype(int)
    features['has_mention'] = df['tweet_clean'].str.contains('MENTION').astype(int)

    # Weather keywords
    weather_keywords = {
        'positive': ['love', 'beautiful', 'perfect', 'awesome', 'great', 'nice', 'enjoy'],
        'negative': ['hate', 'terrible', 'awful', 'bad', 'worst', 'annoying'],
        'hot': ['hot', 'heat', 'warm', 'sweating'],
        'cold': ['cold', 'freeze', 'freezing', 'chilly'],
        'rain': ['rain', 'rainy', 'raining', 'shower'],
        'snow': ['snow', 'snowy', 'snowing', 'blizzard'],
        'wind': ['wind', 'windy', 'breeze', 'gust'],
        'storm': ['storm', 'thunder', 'lightning'],
        'sun': ['sun', 'sunny', 'sunshine'],
        'cloud': ['cloud', 'cloudy', 'overcast'],
        'past': ['yesterday', 'was', 'had', 'earlier'],
        'future': ['tomorrow', 'will', 'forecast', 'predict'],
        'current': ['now', 'today', 'currently', 'right now']
    }

    for keyword_type, keywords in weather_keywords.items():
        pattern = '|'.join(keywords)
        features[f'has_{keyword_type}'] = df['tweet_clean'].str.contains(pattern, case=False, na=False).astype(int)

    return features.fillna(0)

train_text_features = extract_text_features(train)
test_text_features = extract_text_features(test)

# 4. Encode categorical features
le_state = LabelEncoder()
le_location = LabelEncoder()

all_states = pd.concat([train['state'], test['state']])
all_locations = pd.concat([train['location'], test['location']])

le_state.fit(all_states)
le_location.fit(all_locations)

train_state = le_state.transform(train['state']).reshape(-1, 1)
train_location = le_location.transform(train['location']).reshape(-1, 1)
test_state = le_state.transform(test['state']).reshape(-1, 1)
test_location = le_location.transform(test['location']).reshape(-1, 1)

# Combine all features
from scipy.sparse import hstack, csr_matrix
X_train = hstack([
    X_train_tfidf,
    X_train_char,
    csr_matrix(train_text_features.values),
    csr_matrix(train_state),
    csr_matrix(train_location)
])
X_test = hstack([
    X_test_tfidf,
    X_test_char,
    csr_matrix(test_text_features.values),
    csr_matrix(test_state),
    csr_matrix(test_location)
])

y_train = train[all_target_cols].values

print(f"Feature shape: {X_train.shape}")
print(f"Target shape: {y_train.shape}")

# Try XGBoost first, fall back to Ridge if not available
try:
    import xgboost as xgb
    print("\nTraining XGBoost models...")

    # Sentiment model
    print("Training sentiment model...")
    sentiment_preds = np.zeros((len(test), 5))
    for i in range(5):
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist'
        )
        model.fit(X_train, y_train[:, i])
        sentiment_preds[:, i] = model.predict(X_test)

    # When model
    print("Training when model...")
    when_preds = np.zeros((len(test), 4))
    for i in range(4):
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist'
        )
        model.fit(X_train, y_train[:, 5+i])
        when_preds[:, i] = model.predict(X_test)

    # Kind model
    print("Training kind model...")
    kind_preds = np.zeros((len(test), 15))
    for i in range(15):
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist'
        )
        model.fit(X_train, y_train[:, 9+i])
        kind_preds[:, i] = model.predict(X_test)

except ImportError:
    print("\nXGBoost not available, using Ridge regression...")
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor

    sentiment_model = MultiOutputRegressor(Ridge(alpha=0.5))
    sentiment_model.fit(X_train, y_train[:, 0:5])
    sentiment_preds = sentiment_model.predict(X_test)

    when_model = MultiOutputRegressor(Ridge(alpha=0.5))
    when_model.fit(X_train, y_train[:, 5:9])
    when_preds = when_model.predict(X_test)

    kind_model = MultiOutputRegressor(Ridge(alpha=0.5))
    kind_model.fit(X_train, y_train[:, 9:24])
    kind_preds = kind_model.predict(X_test)

# Post-processing
print("\nPost-processing predictions...")
# Sentiment - normalize to sum to 1
sentiment_preds = np.maximum(sentiment_preds, 0)
sentiment_sums = sentiment_preds.sum(axis=1, keepdims=True)
sentiment_sums[sentiment_sums == 0] = 1
sentiment_preds = sentiment_preds / sentiment_sums

# When - normalize to sum to 1
when_preds = np.maximum(when_preds, 0)
when_sums = when_preds.sum(axis=1, keepdims=True)
when_sums[when_sums == 0] = 1
when_preds = when_preds / when_sums

# Kind - clip to [0, 1]
kind_preds = np.clip(kind_preds, 0, 1)

# Combine predictions
predictions = np.hstack([sentiment_preds, when_preds, kind_preds])

# Create submission
print("Creating submission file...")
submission = pd.DataFrame({'id': test['id']})
for i, col in enumerate(all_target_cols):
    submission[col] = predictions[:, i]

submission.to_csv('submission_improved.csv', index=False)
print("\n✓ Submission file created: submission_improved.csv")

# Show sample predictions
print("\nSample predictions:")
print(submission.head(10))
print(f"\nSubmission shape: {submission.shape}")

# Verify
print("\nValidation:")
print(f"Sentiment sums: min={submission[sentiment_cols].sum(axis=1).min():.4f}, max={submission[sentiment_cols].sum(axis=1).max():.4f}")
print(f"When sums: min={submission[when_cols].sum(axis=1).min():.4f}, max={submission[when_cols].sum(axis=1).max():.4f}")
print(f"Kind values: min={submission[kind_cols].min().min():.4f}, max={submission[kind_cols].max().max():.4f}")

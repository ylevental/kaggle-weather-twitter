import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
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

# Enhanced text preprocessing
def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', 'URL', text)
    text = re.sub(r'@\w+', 'MENTION', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    return text

print("Preprocessing text...")
train['tweet_clean'] = train['tweet'].apply(preprocess_text)
test['tweet_clean'] = test['tweet'].apply(preprocess_text)

# Fill missing values
train['state'] = train['state'].fillna('unknown')
test['state'] = test['state'].fillna('unknown')
train['location'] = train['location'].fillna('unknown')
test['location'] = test['location'].fillna('unknown')

# Enhanced feature engineering
print("Creating enhanced features...")

# 1. Multiple TF-IDF representations
tfidf_word = TfidfVectorizer(max_features=2000, ngram_range=(1, 3), min_df=2, max_df=0.9, sublinear_tf=True)
X_train_tfidf_word = tfidf_word.fit_transform(train['tweet_clean'])
X_test_tfidf_word = tfidf_word.transform(test['tweet_clean'])

# Character n-grams
tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=500, min_df=3)
X_train_tfidf_char = tfidf_char.fit_transform(train['tweet_clean'])
X_test_tfidf_char = tfidf_char.transform(test['tweet_clean'])

# 2. Enhanced text statistics
def extract_enhanced_features(df):
    features = pd.DataFrame()

    # Basic stats
    features['tweet_length'] = df['tweet_clean'].str.len()
    features['word_count'] = df['tweet_clean'].str.split().str.len()
    features['avg_word_length'] = features['tweet_length'] / (features['word_count'] + 1)
    features['unique_words'] = df['tweet_clean'].apply(lambda x: len(set(str(x).split())))
    features['unique_ratio'] = features['unique_words'] / (features['word_count'] + 1)

    # Punctuation and style
    features['exclamation_count'] = df['tweet'].str.count('!')
    features['question_count'] = df['tweet'].str.count(r'\?')
    features['period_count'] = df['tweet'].str.count(r'\.')
    features['comma_count'] = df['tweet'].str.count(',')
    features['caps_count'] = df['tweet'].str.count('[A-Z]')
    features['caps_ratio'] = features['caps_count'] / (features['tweet_length'] + 1)
    features['has_url'] = df['tweet_clean'].str.contains('URL').astype(int)
    features['has_mention'] = df['tweet_clean'].str.contains('MENTION').astype(int)
    features['ellipsis_count'] = df['tweet'].str.count(r'\.\.\.')

    # Sentiment words (expanded)
    positive_words = ['love', 'beautiful', 'perfect', 'awesome', 'great', 'nice', 'enjoy',
                      'amazing', 'wonderful', 'excellent', 'fantastic', 'good', 'happy', 'best']
    negative_words = ['hate', 'terrible', 'awful', 'bad', 'worst', 'annoying', 'sucks',
                      'horrible', 'sad', 'depressing', 'miserable', 'hate']

    features['positive_word_count'] = df['tweet_clean'].apply(
        lambda x: sum(1 for w in positive_words if w in str(x))
    )
    features['negative_word_count'] = df['tweet_clean'].apply(
        lambda x: sum(1 for w in negative_words if w in str(x))
    )

    # Weather-specific keywords (expanded)
    weather_keywords = {
        'hot': ['hot', 'heat', 'warm', 'sweating', 'sweat', 'boiling', 'burning'],
        'cold': ['cold', 'freeze', 'freezing', 'chilly', 'frigid', 'icy', 'frozen'],
        'rain': ['rain', 'rainy', 'raining', 'shower', 'drizzle', 'downpour', 'wet'],
        'snow': ['snow', 'snowy', 'snowing', 'blizzard', 'flurries', 'snowfall'],
        'wind': ['wind', 'windy', 'breeze', 'gust', 'breezy', 'gusty'],
        'storm': ['storm', 'thunder', 'lightning', 'thunderstorm', 'stormy'],
        'sun': ['sun', 'sunny', 'sunshine', 'bright', 'clear'],
        'cloud': ['cloud', 'cloudy', 'overcast', 'grey', 'gray'],
        'humid': ['humid', 'humidity', 'muggy', 'sticky'],
        'dry': ['dry', 'arid', 'drought'],
        'tornado': ['tornado', 'twister', 'funnel'],
        'hurricane': ['hurricane', 'tropical storm', 'typhoon'],
        'past': ['yesterday', 'was', 'had', 'earlier', 'ago', 'last', 'previous'],
        'future': ['tomorrow', 'will', 'forecast', 'predict', 'later', 'next', 'gonna', 'going to'],
        'current': ['now', 'today', 'currently', 'right now', 'this', 'present']
    }

    for keyword_type, keywords in weather_keywords.items():
        pattern = '|'.join(keywords)
        features[f'has_{keyword_type}'] = df['tweet_clean'].str.contains(pattern, case=False, na=False).astype(int)
        features[f'count_{keyword_type}'] = df['tweet_clean'].apply(
            lambda x: sum(str(x).lower().count(kw) for kw in keywords)
        )

    return features.fillna(0)

train_features = extract_enhanced_features(train)
test_features = extract_enhanced_features(test)

print(f"Engineered features: {train_features.shape[1]}")

# Encode categorical
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
    X_train_tfidf_word,
    X_train_tfidf_char,
    csr_matrix(train_features.values),
    csr_matrix(train_state),
    csr_matrix(train_location)
])
X_test = hstack([
    X_test_tfidf_word,
    X_test_tfidf_char,
    csr_matrix(test_features.values),
    csr_matrix(test_state),
    csr_matrix(test_location)
])

y_train = train[all_target_cols].values

print(f"Total features: {X_train.shape[1]}")
print(f"Target shape: {y_train.shape}")

# Try LightGBM first, then XGBoost, then Ridge
try:
    import lightgbm as lgb
    print("\nTraining LightGBM models (best performance)...")

    # Sentiment models
    print("Training sentiment models...")
    sentiment_preds = np.zeros((len(test), 5))
    for i in range(5):
        print(f"  Model {i+1}/5 (s{i+1})...")
        model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train[:, i])
        sentiment_preds[:, i] = model.predict(X_test)
        print(f"  ✓ Completed {i+1}/5")

    # When models
    print("\nTraining when models...")
    when_preds = np.zeros((len(test), 4))
    for i in range(4):
        print(f"  Model {i+1}/4 (w{i+1})...")
        model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train[:, 5+i])
        when_preds[:, i] = model.predict(X_test)
        print(f"  ✓ Completed {i+1}/4")

    # Kind models
    print("\nTraining kind models...")
    kind_preds = np.zeros((len(test), 15))
    for i in range(15):
        print(f"  Model {i+1}/15 (k{i+1})...")
        model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train[:, 9+i])
        kind_preds[:, i] = model.predict(X_test)
        print(f"  ✓ Completed {i+1}/15")

except ImportError:
    try:
        import xgboost as xgb
        print("\nLightGBM not available, using XGBoost...")

        sentiment_preds = np.zeros((len(test), 5))
        for i in range(5):
            print(f"  Sentiment model {i+1}/5...")
            model = xgb.XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42)
            model.fit(X_train, y_train[:, i])
            sentiment_preds[:, i] = model.predict(X_test)

        when_preds = np.zeros((len(test), 4))
        for i in range(4):
            print(f"  When model {i+1}/4...")
            model = xgb.XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42)
            model.fit(X_train, y_train[:, 5+i])
            when_preds[:, i] = model.predict(X_test)

        kind_preds = np.zeros((len(test), 15))
        for i in range(15):
            print(f"  Kind model {i+1}/15...")
            model = xgb.XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42)
            model.fit(X_train, y_train[:, 9+i])
            kind_preds[:, i] = model.predict(X_test)

    except ImportError:
        print("\nNeither LightGBM nor XGBoost available, using Ridge...")
        sentiment_model = MultiOutputRegressor(Ridge(alpha=0.3))
        sentiment_model.fit(X_train, y_train[:, 0:5])
        sentiment_preds = sentiment_model.predict(X_test)

        when_model = MultiOutputRegressor(Ridge(alpha=0.3))
        when_model.fit(X_train, y_train[:, 5:9])
        when_preds = when_model.predict(X_test)

        kind_model = MultiOutputRegressor(Ridge(alpha=0.3))
        kind_model.fit(X_train, y_train[:, 9:24])
        kind_preds = kind_model.predict(X_test)

# Post-processing
print("\nPost-processing predictions...")
sentiment_preds = np.maximum(sentiment_preds, 0)
sentiment_sums = sentiment_preds.sum(axis=1, keepdims=True)
sentiment_sums[sentiment_sums == 0] = 1
sentiment_preds = sentiment_preds / sentiment_sums

when_preds = np.maximum(when_preds, 0)
when_sums = when_preds.sum(axis=1, keepdims=True)
when_sums[when_sums == 0] = 1
when_preds = when_preds / when_sums

kind_preds = np.clip(kind_preds, 0, 1)

predictions = np.hstack([sentiment_preds, when_preds, kind_preds])

# Create submission
print("Creating submission file...")
submission = pd.DataFrame({'id': test['id']})
for i, col in enumerate(all_target_cols):
    submission[col] = predictions[:, i]

submission.to_csv('submission_best.csv', index=False)
print("\n✓ Submission file created: submission_best.csv")

print("\nSample predictions:")
print(submission.head(10))
print(f"\nSubmission shape: {submission.shape}")

print("\nValidation:")
print(f"Sentiment sums: min={submission[sentiment_cols].sum(axis=1).min():.4f}, max={submission[sentiment_cols].sum(axis=1).max():.4f}")
print(f"When sums: min={submission[when_cols].sum(axis=1).min():.4f}, max={submission[when_cols].sum(axis=1).max():.4f}")
print(f"Kind values: min={submission[kind_cols].min().min():.4f}, max={submission[kind_cols].max().max():.4f}")

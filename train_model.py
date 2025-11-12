import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
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

# Fill missing values
train['tweet'] = train['tweet'].fillna('')
test['tweet'] = test['tweet'].fillna('')
train['state'] = train['state'].fillna('unknown')
test['state'] = test['state'].fillna('unknown')
train['location'] = train['location'].fillna('unknown')
test['location'] = test['location'].fillna('unknown')

# Feature engineering
print("Creating features...")

# Text features using TF-IDF with reduced features
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=3, max_df=0.9)
X_train_text = tfidf.fit_transform(train['tweet'])
X_test_text = tfidf.transform(test['tweet'])

print(f"TF-IDF feature shape: {X_train_text.shape}")

# Encode categorical features
le_state = LabelEncoder()
le_location = LabelEncoder()

# Fit on combined data to handle unseen categories
all_states = pd.concat([train['state'], test['state']])
all_locations = pd.concat([train['location'], test['location']])

le_state.fit(all_states)
le_location.fit(all_locations)

train_state = le_state.transform(train['state']).reshape(-1, 1)
train_location = le_location.transform(train['location']).reshape(-1, 1)
test_state = le_state.transform(test['state']).reshape(-1, 1)
test_location = le_location.transform(test['location']).reshape(-1, 1)

# Combine features - use sparse matrices
from scipy.sparse import hstack, csr_matrix
X_train = hstack([X_train_text, csr_matrix(train_state), csr_matrix(train_location)])
X_test = hstack([X_test_text, csr_matrix(test_state), csr_matrix(test_location)])

# Target
y_train = train[all_target_cols].values

print(f"Feature shape: {X_train.shape}")
print(f"Target shape: {y_train.shape}")

# Train separate models for each category group to save memory
print("\nTraining models...")

# Sentiment model (s1-s5)
print("Training sentiment model...")
sentiment_model = MultiOutputRegressor(Ridge(alpha=1.0))
sentiment_model.fit(X_train, y_train[:, 0:5])
sentiment_preds = sentiment_model.predict(X_test)

# When model (w1-w4)
print("Training when model...")
when_model = MultiOutputRegressor(Ridge(alpha=1.0))
when_model.fit(X_train, y_train[:, 5:9])
when_preds = when_model.predict(X_test)

# Kind model (k1-k15)
print("Training kind model...")
kind_model = MultiOutputRegressor(Ridge(alpha=1.0))
kind_model.fit(X_train, y_train[:, 9:24])
kind_preds = kind_model.predict(X_test)

# Combine predictions
predictions = np.hstack([sentiment_preds, when_preds, kind_preds])

# Post-processing: ensure sentiment and when categories sum to 1
print("\nPost-processing predictions...")
# For sentiment columns (0-4)
sentiment_preds = np.maximum(sentiment_preds, 0)  # No negative values
sentiment_sums = sentiment_preds.sum(axis=1, keepdims=True)
sentiment_sums[sentiment_sums == 0] = 1  # Avoid division by zero
sentiment_preds = sentiment_preds / sentiment_sums

# For when columns (5-8)
when_preds = np.maximum(when_preds, 0)
when_sums = when_preds.sum(axis=1, keepdims=True)
when_sums[when_sums == 0] = 1
when_preds = when_preds / when_sums

# For kind columns (9-23) - just clip to [0, 1]
kind_preds = np.clip(kind_preds, 0, 1)

# Combine normalized predictions
predictions = np.hstack([sentiment_preds, when_preds, kind_preds])

# Create submission
print("Creating submission file...")
submission = pd.DataFrame({
    'id': test['id']
})

for i, col in enumerate(all_target_cols):
    submission[col] = predictions[:, i]

submission.to_csv('submission.csv', index=False)
print("\n✓ Submission file created: submission.csv")

# Show sample predictions
print("\nSample predictions:")
print(submission.head(10))
print(f"\nSubmission shape: {submission.shape}")
print(f"Expected shape: ({len(test)}, 25)")  # 1 id + 24 targets

# Verify sums for sentiment and when columns
print("\nValidation:")
print(f"Sentiment sums (should be ~1.0): min={submission[sentiment_cols].sum(axis=1).min():.4f}, max={submission[sentiment_cols].sum(axis=1).max():.4f}")
print(f"When sums (should be ~1.0): min={submission[when_cols].sum(axis=1).min():.4f}, max={submission[when_cols].sum(axis=1).max():.4f}")
print(f"Kind values: min={submission[kind_cols].min().min():.4f}, max={submission[kind_cols].max().max():.4f}")

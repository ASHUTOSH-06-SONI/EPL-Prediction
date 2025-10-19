# train_and_save_model.py

import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
MODEL_FILE = 'epl_predictor_model.pkl'
ENCODER_FILE = 'team_encoder.pkl'
DATA_FILE = 'df_processed.pkl'
FEATURES_FILE = 'available_features.pkl'
TUNING_MODE = "FAST" 
# ---------------------

# Load dataset
try:
    df = pd.read_csv('epl_all.csv')
except FileNotFoundError:
    print("FATAL ERROR: 'epl_all.csv' not found. Please ensure it is in the same directory.")
    exit()

print("="*70)
print(" "*20 + "ENHANCED EPL PREDICTOR TRAINING")
print("="*70)
print(f"\nDataset shape: {df.shape}")

# ============================================================================
# ADVANCED FEATURE ENGINEERING (unchanged)
# ============================================================================
def advanced_feature_engineering(df):
    """
    Create comprehensive features from FBRef data
    """
    df = df.copy()
    
    # Convert date
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Encode result
    le_result = LabelEncoder()
    df['result_encoded'] = le_result.fit_transform(df['FTR'])
    
    # Team encoding
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    team_encoder = LabelEncoder()
    team_encoder.fit(all_teams)
    
    df['home_team_encoded'] = team_encoder.transform(df['HomeTeam'])
    df['away_team_encoded'] = team_encoder.transform(df['AwayTeam'])
    
    # Initialize feature columns
    feature_cols = [
        'home_goals_scored_5', 'away_goals_scored_5',
        'home_goals_conceded_5', 'away_goals_conceded_5',
        'home_shots_5', 'away_shots_5',
        'home_shots_target_5', 'away_shots_target_5',
        'home_fouls_5', 'away_fouls_5',
        'home_corners_5', 'away_corners_5',
        'home_yellows_5', 'away_yellows_5',
        'home_form_5', 'away_form_5',
        'home_win_pct_5', 'away_win_pct_5',
        'home_goals_scored_10', 'away_goals_scored_10',
        'home_goals_conceded_10', 'away_goals_conceded_10',
        'home_form_10', 'away_form_10'
    ]
    
    for col in feature_cols:
        df[col] = 0.0
    
    print("\nCalculating rolling statistics for each team...")
    
    # Calculate rolling stats for each team
    for team in all_teams:
        # HOME GAMES
        home_mask = df['HomeTeam'] == team
        home_games = df[home_mask].copy()
        
        if len(home_games) > 0:
            # Shift(1).fillna(0) is crucial to avoid data leakage
            home_games['home_goals_scored_5'] = home_games['FTHG'].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            home_games['home_goals_conceded_5'] = home_games['FTAG'].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            home_games['home_shots_5'] = home_games['HS'].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            home_games['home_shots_target_5'] = home_games['HST'].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            home_games['home_fouls_5'] = home_games['HF'].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            home_games['home_corners_5'] = home_games['HC'].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            home_games['home_yellows_5'] = home_games['HY'].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            
            home_games['home_points'] = home_games['FTR'].map({'H': 3, 'D': 1, 'A': 0})
            home_games['home_form_5'] = home_games['home_points'].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            home_games['home_wins'] = (home_games['FTR'] == 'H').astype(int)
            home_games['home_win_pct_5'] = home_games['home_wins'].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            
            home_games['home_goals_scored_10'] = home_games['FTHG'].rolling(10, min_periods=1).mean().shift(1).fillna(0)
            home_games['home_goals_conceded_10'] = home_games['FTAG'].rolling(10, min_periods=1).mean().shift(1).fillna(0)
            home_games['home_form_10'] = home_games['home_points'].rolling(10, min_periods=1).mean().shift(1).fillna(0)
            
            for col in [c for c in feature_cols if c.startswith('home_')]:
                if col in home_games.columns:
                    df.loc[home_mask, col] = home_games[col].values
        
        # AWAY GAMES
        away_mask = df['AwayTeam'] == team
        away_games = df[away_mask].copy()
        
        if len(away_games) > 0:
            away_games['away_goals_scored_5'] = away_games['FTAG'].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            away_games['away_goals_conceded_5'] = away_games['FTHG'].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            away_games['away_shots_5'] = away_games['AS'].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            away_games['away_shots_target_5'] = away_games['AST'].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            away_games['away_fouls_5'] = away_games['AF'].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            away_games['away_corners_5'] = away_games['AC'].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            away_games['away_yellows_5'] = away_games['AY'].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            
            away_games['away_points'] = away_games['FTR'].map({'A': 3, 'D': 1, 'H': 0})
            away_games['away_form_5'] = away_games['away_points'].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            away_games['away_wins'] = (away_games['FTR'] == 'A').astype(int)
            away_games['away_win_pct_5'] = away_games['away_wins'].rolling(5, min_periods=1).mean().shift(1).fillna(0)
            
            away_games['away_goals_scored_10'] = away_games['FTAG'].rolling(10, min_periods=1).mean().shift(1).fillna(0)
            away_games['away_goals_conceded_10'] = away_games['FTHG'].rolling(10, min_periods=1).mean().shift(1).fillna(0)
            away_games['away_form_10'] = away_games['away_points'].rolling(10, min_periods=1).mean().shift(1).fillna(0)
            
            for col in [c for c in feature_cols if c.startswith('away_')]:
                if col in away_games.columns:
                    df.loc[away_mask, col] = away_games[col].values
    
    # DERIVED FEATURES - Differential metrics
    print("Creating differential features...")
    
    df['goal_diff_5'] = df['home_goals_scored_5'] - df['away_goals_scored_5']
    df['defense_diff_5'] = df['away_goals_conceded_5'] - df['home_goals_conceded_5']
    df['form_diff_5'] = df['home_form_5'] - df['away_form_5']
    df['shots_diff_5'] = df['home_shots_5'] - df['away_shots_5']
    df['shots_target_diff_5'] = df['home_shots_target_5'] - df['away_shots_target_5']
    df['corners_diff_5'] = df['home_corners_5'] - df['away_corners_5']
    df['discipline_diff_5'] = df['away_yellows_5'] - df['home_yellows_5']
    
    # Attacking/Defensive strength ratios
    df['home_attack_strength'] = df['home_goals_scored_5'] / (df['home_goals_conceded_5'] + 0.1)
    df['away_attack_strength'] = df['away_goals_scored_5'] / (df['away_goals_conceded_5'] + 0.1)
    
    # Shot accuracy
    df['home_shot_accuracy'] = df['home_shots_target_5'] / (df['home_shots_5'] + 0.1)
    df['away_shot_accuracy'] = df['away_shots_target_5'] / (df['away_shots_5'] + 0.1)
    
    # Consistency metrics (longer window)
    df['home_consistency'] = df['home_form_10'] - df['home_form_5']
    df['away_consistency'] = df['away_form_10'] - df['away_form_5']
    
    # Add betting odds if available (strong predictor!)
    if 'B365H' in df.columns and 'B365D' in df.columns and 'B365A' in df.columns:
        df['odds_home'] = df['B365H']
        df['odds_draw'] = df['B365D']
        df['odds_away'] = df['B365A']
        
        df['implied_home_prob'] = 1 / df['odds_home']
        df['implied_draw_prob'] = 1 / df['odds_draw']
        df['implied_away_prob'] = 1 / df['odds_away']
        
        total = df['implied_home_prob'] + df['implied_draw_prob'] + df['implied_away_prob']
        df['norm_home_prob'] = df['implied_home_prob'] / total
        df['norm_draw_prob'] = df['implied_draw_prob'] / total
        df['norm_away_prob'] = df['implied_away_prob'] / total
    
    df = df.fillna(0) 
    
    return df, team_encoder, le_result

# Process data
print("\nProcessing features...")
df_processed, team_encoder, le_result = advanced_feature_engineering(df)

# ============================================================================
# FEATURE SELECTION (unchanged)
# ============================================================================
basic_features = [
    'home_team_encoded', 'away_team_encoded',
    'home_goals_scored_5', 'away_goals_scored_5',
    'home_goals_conceded_5', 'away_goals_conceded_5',
    'home_form_5', 'away_form_5'
]

advanced_features = [
    'home_shots_5', 'away_shots_5',
    'home_shots_target_5', 'away_shots_target_5',
    'home_corners_5', 'away_corners_5',
    'home_yellows_5', 'away_yellows_5',
    'home_win_pct_5', 'away_win_pct_5',
    'goal_diff_5', 'defense_diff_5', 'form_diff_5',
    'shots_diff_5', 'shots_target_diff_5', 'corners_diff_5',
    'home_attack_strength', 'away_attack_strength',
    'home_shot_accuracy', 'away_shot_accuracy',
    'home_consistency', 'away_consistency'
]

odds_features = [
    'norm_home_prob', 'norm_draw_prob', 'norm_away_prob'
]

all_features = basic_features + advanced_features
if 'norm_home_prob' in df_processed.columns:
    all_features += odds_features
    print("\n✓ Including betting odds features")

available_features = [f for f in all_features if f in df_processed.columns]
print(f"\nTotal features: {len(available_features)}")

X = df_processed[available_features]
y = df_processed['result_encoded']

valid_idx = df_processed.index[10:]
X = X.loc[valid_idx]
y = y.loc[valid_idx]

print(f"Samples after filtering: {len(X)}")

# ============================================================================
# TRAIN/TEST SPLIT - TIME-BASED (unchanged)
# ============================================================================
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nTrain set: {len(X_train)} matches")
print(f"Test set: {len(X_test)} matches")

# ============================================================================
# HYPERPARAMETER TUNING (FIXED MODE) (unchanged)
# ============================================================================
print("\n" + "="*70)
print(f"HYPERPARAMETER TUNING ({TUNING_MODE} Mode)")
print("="*70)

if TUNING_MODE == "FAST":
    param_grid = {
        'n_estimators': [200],
        'max_depth': [4, 5],
        'learning_rate': [0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_weight': [3],
        'gamma': [0, 0.1]
    }
    cv_folds = 2
    
    total_combinations = 1
    for param_values in param_grid.values():
        total_combinations *= len(param_values)
    
    print(f"\nTesting {total_combinations} parameter combinations with {cv_folds}-fold CV")
    
    base_model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    best_model = grid_search.best_estimator_

else:
    print("\n✓ Using default parameters (no grid search)")
    best_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    best_model.fit(X_train, y_train)


print("\n✓ Model training complete!")

# ============================================================================
# EVALUATION (Simplified for console output)
# ============================================================================
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n" + "="*70)
print("MODEL EVALUATION")
print(f"📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("="*70)

# ============================================================================
# SAVE ARTIFACTS - FIX APPLIED HERE
# ============================================================================

# Save the model
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(best_model, f)
print(f"💾 Model saved to {MODEL_FILE}")

# Save the team encoder
with open(ENCODER_FILE, 'wb') as f:
    pickle.dump(team_encoder, f)
print(f"💾 Team Encoder saved to {ENCODER_FILE}")

# Save the processed dataframe (for feature lookup/form calculation)
# 🚨 FIX: Explicitly include 'FTHG' and 'FTAG' for H2H display in Streamlit
REQUIRED_H2H_COLS = ['FTHG', 'FTAG'] 

df_save = df_processed[['Date', 'HomeTeam', 'AwayTeam', 'FTR'] + REQUIRED_H2H_COLS + \
                      [f for f in available_features if f in df_processed.columns]].tail(500) 
                      
df_save.to_pickle(DATA_FILE)
print(f"💾 Processed Data saved to {DATA_FILE} (Including FTHG, FTAG for H2H)")

# Save the list of features
with open(FEATURES_FILE, 'wb') as f:
    pickle.dump(available_features, f)
print(f"💾 Available Features saved to {FEATURES_FILE}")

print("\n✨ Deployment preparation complete! Run 'streamlit run app.py'")

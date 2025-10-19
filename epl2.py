import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('epl_all.csv')

print("="*70)
print(" "*20 + "ENHANCED EPL PREDICTOR")
print("="*70)
print(f"\nDataset shape: {df.shape}")

# ============================================================================
# ADVANCED FEATURE ENGINEERING
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
            # Last 5 home games
            home_games['home_goals_scored_5'] = home_games['FTHG'].rolling(5, min_periods=1).mean()
            home_games['home_goals_conceded_5'] = home_games['FTAG'].rolling(5, min_periods=1).mean()
            home_games['home_shots_5'] = home_games['HS'].rolling(5, min_periods=1).mean()
            home_games['home_shots_target_5'] = home_games['HST'].rolling(5, min_periods=1).mean()
            home_games['home_fouls_5'] = home_games['HF'].rolling(5, min_periods=1).mean()
            home_games['home_corners_5'] = home_games['HC'].rolling(5, min_periods=1).mean()
            home_games['home_yellows_5'] = home_games['HY'].rolling(5, min_periods=1).mean()
            
            # Points calculation
            home_games['home_points'] = home_games['FTR'].map({'H': 3, 'D': 1, 'A': 0})
            home_games['home_form_5'] = home_games['home_points'].rolling(5, min_periods=1).mean()
            home_games['home_wins'] = (home_games['FTR'] == 'H').astype(int)
            home_games['home_win_pct_5'] = home_games['home_wins'].rolling(5, min_periods=1).mean()
            
            # Last 10 home games
            home_games['home_goals_scored_10'] = home_games['FTHG'].rolling(10, min_periods=1).mean()
            home_games['home_goals_conceded_10'] = home_games['FTAG'].rolling(10, min_periods=1).mean()
            home_games['home_form_10'] = home_games['home_points'].rolling(10, min_periods=1).mean()
            
            # Update main dataframe
            for col in [c for c in feature_cols if c.startswith('home_')]:
                if col in home_games.columns:
                    df.loc[home_mask, col] = home_games[col].values
        
        # AWAY GAMES
        away_mask = df['AwayTeam'] == team
        away_games = df[away_mask].copy()
        
        if len(away_games) > 0:
            # Last 5 away games
            away_games['away_goals_scored_5'] = away_games['FTAG'].rolling(5, min_periods=1).mean()
            away_games['away_goals_conceded_5'] = away_games['FTHG'].rolling(5, min_periods=1).mean()
            away_games['away_shots_5'] = away_games['AS'].rolling(5, min_periods=1).mean()
            away_games['away_shots_target_5'] = away_games['AST'].rolling(5, min_periods=1).mean()
            away_games['away_fouls_5'] = away_games['AF'].rolling(5, min_periods=1).mean()
            away_games['away_corners_5'] = away_games['AC'].rolling(5, min_periods=1).mean()
            away_games['away_yellows_5'] = away_games['AY'].rolling(5, min_periods=1).mean()
            
            # Points calculation
            away_games['away_points'] = away_games['FTR'].map({'A': 3, 'D': 1, 'H': 0})
            away_games['away_form_5'] = away_games['away_points'].rolling(5, min_periods=1).mean()
            away_games['away_wins'] = (away_games['FTR'] == 'A').astype(int)
            away_games['away_win_pct_5'] = away_games['away_wins'].rolling(5, min_periods=1).mean()
            
            # Last 10 away games
            away_games['away_goals_scored_10'] = away_games['FTAG'].rolling(10, min_periods=1).mean()
            away_games['away_goals_conceded_10'] = away_games['FTHG'].rolling(10, min_periods=1).mean()
            away_games['away_form_10'] = away_games['away_points'].rolling(10, min_periods=1).mean()
            
            # Update main dataframe
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
    df['discipline_diff_5'] = df['away_yellows_5'] - df['home_yellows_5']  # Higher away yellows is advantage home
    
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
        
        # Implied probabilities (bookmaker's assessment)
        df['implied_home_prob'] = 1 / df['odds_home']
        df['implied_draw_prob'] = 1 / df['odds_draw']
        df['implied_away_prob'] = 1 / df['odds_away']
        
        # Normalize probabilities
        total = df['implied_home_prob'] + df['implied_draw_prob'] + df['implied_away_prob']
        df['norm_home_prob'] = df['implied_home_prob'] / total
        df['norm_draw_prob'] = df['implied_draw_prob'] / total
        df['norm_away_prob'] = df['implied_away_prob'] / total
    
    # Fill any remaining NaN
    df = df.fillna(method='bfill').fillna(0)
    
    return df, team_encoder, le_result

# Process data
print("\nProcessing features...")
df_processed, team_encoder, le_result = advanced_feature_engineering(df)

# ============================================================================
# FEATURE SELECTION
# ============================================================================

# Define feature sets
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

# Combine features
all_features = basic_features + advanced_features
if 'norm_home_prob' in df_processed.columns:
    all_features += odds_features
    print("\n✓ Including betting odds features")

# Filter features that exist
available_features = [f for f in all_features if f in df_processed.columns]
print(f"\nTotal features: {len(available_features)}")

X = df_processed[available_features]
y = df_processed['result_encoded']

# Remove rows with insufficient historical data (first 10 games per team)
valid_idx = df_processed.index[10:]
X = X.loc[valid_idx]
y = y.loc[valid_idx]

print(f"Samples after filtering: {len(X)}")

# ============================================================================
# TRAIN/TEST SPLIT - TIME-BASED
# ============================================================================

# Time-based split (more realistic for sports prediction)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nTrain set: {len(X_train)} matches")
print(f"Test set: {len(X_test)} matches")

# ============================================================================
# HYPERPARAMETER TUNING (OPTIMIZED FOR SPEED)
# ============================================================================

print("\n" + "="*70)
print("HYPERPARAMETER TUNING (Laptop-Friendly Mode)")
print("="*70)

# OPTION 1: Quick tuning with fewer parameters (RECOMMENDED)
print("\nChoose tuning mode:")
print("  1. FAST - Quick tuning, ~2 minutes (Recommended)")
print("  2. BALANCED - Moderate tuning, ~5 minutes")
print("  3. THOROUGH - Extensive tuning, ~15+ minutes (Original)")
print("  4. SKIP - Use default parameters, instant")

tuning_choice = input("\nEnter choice (1-4) [default: 1]: ").strip() or "1"

if tuning_choice == "4":
    print("\n✓ Using default parameters (skipping tuning)")
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
    
elif tuning_choice == "1":
    # FAST mode - minimal grid
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
    
elif tuning_choice == "2":
    # BALANCED mode
    param_grid = {
        'n_estimators': [150, 200],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.08, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_weight': [3],
        'gamma': [0, 0.1]
    }
    cv_folds = 3
    
else:  # "3" or anything else - THOROUGH mode
    # Original extensive grid
    param_grid = {
        'n_estimators': [150, 200, 250],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    cv_folds = 3

if tuning_choice != "4":
    # Calculate total combinations
    total_combinations = 1
    for param_values in param_grid.values():
        total_combinations *= len(param_values)
    
    print(f"\nTesting {total_combinations} parameter combinations with {cv_folds}-fold CV")
    print(f"Total fits: {total_combinations * cv_folds}")
    
    # Create base model
    base_model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1  # Use all CPU cores
    )
    
    print("\nSearching for best parameters...")
    print("(Go grab a coffee ☕ - your laptop is working hard!)\n")
    
    # Grid search with progress
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2  # Show progress
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest CV score: {grid_search.best_score_:.4f}")
    
    # Use best model
    best_model = grid_search.best_estimator_

print("\n✓ Model training complete!")

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

# Predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Baseline comparison
baseline_accuracy = y_test.value_counts().max() / len(y_test)
print(f"📊 Baseline (most common): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"📈 Improvement: +{(accuracy - baseline_accuracy)*100:.2f}%")

# Classification report
print("\nDetailed Classification Report:")
target_names = ['Away Win', 'Draw', 'Home Win']
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\n       Predicted:")
print("         A    D    H")
print("Actual A", cm[0])
print("       D", cm[1])
print("       H", cm[2])

# Feature importance
print("\n" + "="*70)
print("TOP 15 MOST IMPORTANT FEATURES")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(15).iterrows():
    print(f"{row['feature']:30s}: {row['importance']:.4f}")

# ============================================================================
# PREDICTION CONFIDENCE ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("PREDICTION CONFIDENCE ANALYSIS")
print("="*70)

# Analyze predictions by confidence level
max_probs = y_pred_proba.max(axis=1)
confidence_levels = pd.qcut(max_probs, q=4, labels=['Low', 'Medium', 'High', 'Very High'])

print("\nAccuracy by confidence level:")
for level in ['Low', 'Medium', 'High', 'Very High']:
    mask = confidence_levels == level
    if mask.sum() > 0:
        level_accuracy = accuracy_score(y_test[mask], y_pred[mask])
        print(f"  {level:12s}: {level_accuracy:.4f} ({mask.sum()} predictions)")

# ============================================================================
# SAVE MODEL AND UTILITIES
# ============================================================================

# Prediction function
def predict_match_enhanced(home_team, away_team, model, team_encoder, df_features):
    """Enhanced match prediction with confidence levels"""
    try:
        recent_home = df_features[df_features['HomeTeam'] == home_team].tail(1)
        recent_away = df_features[df_features['AwayTeam'] == away_team].tail(1)
        
        if len(recent_home) == 0 or len(recent_away) == 0:
            print("❌ Team not found in dataset")
            return None
        
        # Create feature dictionary with all required features
        match_data = {}
        
        # Team encodings
        match_data['home_team_encoded'] = team_encoder.transform([home_team])[0]
        match_data['away_team_encoded'] = team_encoder.transform([away_team])[0]
        
        # Add all features from the recent matches
        for feat in available_features:
            if feat in ['home_team_encoded', 'away_team_encoded']:
                continue  # Already added
            
            if feat.startswith('home_') and feat in recent_home.columns:
                match_data[feat] = recent_home[feat].values[0]
            elif feat.startswith('away_') and feat in recent_away.columns:
                match_data[feat] = recent_away[feat].values[0]
            elif feat.startswith('norm_') and feat in recent_home.columns:
                # Betting odds features from home match perspective
                match_data[feat] = recent_home[feat].values[0]
            elif feat in recent_home.columns:
                # For differential and other calculated features
                match_data[feat] = recent_home[feat].values[0]
        
        # Calculate differential features if not present
        if 'goal_diff_5' in available_features and 'goal_diff_5' not in match_data:
            match_data['goal_diff_5'] = match_data.get('home_goals_scored_5', 0) - match_data.get('away_goals_scored_5', 0)
        
        if 'defense_diff_5' in available_features and 'defense_diff_5' not in match_data:
            match_data['defense_diff_5'] = match_data.get('away_goals_conceded_5', 0) - match_data.get('home_goals_conceded_5', 0)
        
        if 'form_diff_5' in available_features and 'form_diff_5' not in match_data:
            match_data['form_diff_5'] = match_data.get('home_form_5', 0) - match_data.get('away_form_5', 0)
        
        if 'shots_diff_5' in available_features and 'shots_diff_5' not in match_data:
            match_data['shots_diff_5'] = match_data.get('home_shots_5', 0) - match_data.get('away_shots_5', 0)
        
        if 'shots_target_diff_5' in available_features and 'shots_target_diff_5' not in match_data:
            match_data['shots_target_diff_5'] = match_data.get('home_shots_target_5', 0) - match_data.get('away_shots_target_5', 0)
        
        if 'corners_diff_5' in available_features and 'corners_diff_5' not in match_data:
            match_data['corners_diff_5'] = match_data.get('home_corners_5', 0) - match_data.get('away_corners_5', 0)
        
        if 'discipline_diff_5' in available_features and 'discipline_diff_5' not in match_data:
            match_data['discipline_diff_5'] = match_data.get('away_yellows_5', 0) - match_data.get('home_yellows_5', 0)
        
        # Ensure all features are present in correct order
        match_features = pd.DataFrame([match_data], columns=available_features)
        
        # Fill any missing values with 0
        match_features = match_features.fillna(0)
        
        prediction = model.predict(match_features)[0]
        probabilities = model.predict_proba(match_features)[0]
        
        outcomes = ['Away Win', 'Draw', 'Home Win']
        max_prob = probabilities.max()
        
        # Confidence assessment
        if max_prob > 0.6:
            confidence = "Very High"
        elif max_prob > 0.5:
            confidence = "High"
        elif max_prob > 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        print(f"\n{'='*60}")
        print(f"Match Prediction: {home_team} vs {away_team}")
        print(f"{'='*60}")
        print(f"Predicted: {outcomes[prediction]}")
        print(f"Confidence: {confidence} ({max_prob:.1%})")
        print(f"\nProbabilities:")
        for outcome, prob in zip(outcomes, probabilities):
            bar = '█' * int(prob * 50)
            print(f"  {outcome:12s}: {prob:6.1%} {bar}")
        
        return prediction, probabilities, confidence
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

print("\n" + "="*70)
print("✓ Model training complete!")
print("="*70)

# Store in global scope for interactive use
globals()['best_model'] = best_model
globals()['team_encoder'] = team_encoder
globals()['df_processed'] = df_processed
globals()['available_features'] = available_features
globals()['predict_match_enhanced'] = predict_match_enhanced

# ============================================================================
# INTERACTIVE PREDICTION INTERFACE
# ============================================================================

def interactive_prediction_enhanced():
    """
    Enhanced interactive interface with detailed predictions
    """
    print("\n" + "="*70)
    print(" "*18 + "⚽ EPL MATCH PREDICTOR ⚽")
    print("="*70)
    
    # Get available teams
    available_teams = sorted(team_encoder.classes_)
    
    print(f"\n📋 Available Teams ({len(available_teams)}):\n")
    
    # Display in columns for better readability
    cols = 3
    for i in range(0, len(available_teams), cols):
        row_teams = available_teams[i:i+cols]
        for j, team in enumerate(row_teams):
            print(f"  {i+j+1:2d}. {team:25s}", end="")
        print()
    
    print("\n" + "-"*70)
    
    # Helper function to get team input
    def get_team_input(prompt, exclude_team=None):
        while True:
            team_input = input(prompt).strip()
            
            # Check if input is a number
            if team_input.isdigit():
                idx = int(team_input) - 1
                if 0 <= idx < len(available_teams):
                    team = available_teams[idx]
                    if team == exclude_team:
                        print("❌ Cannot select the same team twice!")
                        continue
                    return team
                else:
                    print(f"❌ Please enter a number between 1 and {len(available_teams)}")
            else:
                # Try case-insensitive match
                matches = [t for t in available_teams if t.lower() == team_input.lower()]
                if matches:
                    team = matches[0]
                    if team == exclude_team:
                        print("❌ Cannot select the same team twice!")
                        continue
                    return team
                # Try partial match
                partial_matches = [t for t in available_teams if team_input.lower() in t.lower()]
                if len(partial_matches) == 1:
                    team = partial_matches[0]
                    if team == exclude_team:
                        print("❌ Cannot select the same team twice!")
                        continue
                    print(f"✓ Selected: {team}")
                    return team
                elif len(partial_matches) > 1:
                    print(f"❌ Multiple matches found: {', '.join(partial_matches)}")
                    print("   Please be more specific")
                else:
                    print(f"❌ Team not found. Please try again.")
    
    # Get first team
    team1 = get_team_input("\n🏟️  Enter FIRST team (name or number): ")
    
    # Get home/away for first team
    while True:
        location = input(f"\n📍 Is {team1} playing at (H)ome or (A)way? ").strip().upper()
        if location in ['H', 'HOME']:
            home_team = team1
            print(f"\n✓ {team1} will play at HOME")
            break
        elif location in ['A', 'AWAY']:
            away_team = team1
            print(f"\n✓ {team1} will play AWAY")
            break
        else:
            print("❌ Please enter 'H' for Home or 'A' for Away")
    
    # Get second team
    print("\n" + "-"*70)
    team2 = get_team_input(f"\n🏟️  Enter SECOND team (name or number): ", exclude_team=team1)
    
    # Set the other team's location
    if location in ['H', 'HOME']:
        away_team = team2
        print(f"\n✓ {team2} will play AWAY")
    else:
        home_team = team2
        print(f"\n✓ {team2} will play at HOME")
    
    # Display recent form before prediction
    print("\n" + "="*70)
    print("📊 RECENT FORM ANALYSIS")
    print("="*70)
    
    # Get recent home team stats
    recent_home = df_processed[df_processed['HomeTeam'] == home_team].tail(1)
    recent_away = df_processed[df_processed['AwayTeam'] == away_team].tail(1)
    
    if len(recent_home) > 0:
        print(f"\n🏠 {home_team} (Home):")
        print(f"   Goals scored (last 5 home):  {recent_home['home_goals_scored_5'].values[0]:.2f} per game")
        print(f"   Goals conceded (last 5 home): {recent_home['home_goals_conceded_5'].values[0]:.2f} per game")
        print(f"   Form points (last 5 home):    {recent_home['home_form_5'].values[0]:.2f} / 3.00")
        if 'home_win_pct_5' in recent_home.columns:
            print(f"   Win rate (last 5 home):       {recent_home['home_win_pct_5'].values[0]*100:.1f}%")
    
    if len(recent_away) > 0:
        print(f"\n✈️  {away_team} (Away):")
        print(f"   Goals scored (last 5 away):   {recent_away['away_goals_scored_5'].values[0]:.2f} per game")
        print(f"   Goals conceded (last 5 away): {recent_away['away_goals_conceded_5'].values[0]:.2f} per game")
        print(f"   Form points (last 5 away):    {recent_away['away_form_5'].values[0]:.2f} / 3.00")
        if 'away_win_pct_5' in recent_away.columns:
            print(f"   Win rate (last 5 away):       {recent_away['away_win_pct_5'].values[0]*100:.1f}%")
    
    # Generate prediction
    print("\n" + "="*70)
    print("🔮 GENERATING PREDICTION...")
    print("="*70)
    
    result = predict_match_enhanced(home_team, away_team, best_model, team_encoder, df_processed)
    
    if result:
        prediction, probabilities, confidence = result
        
        # Additional insights
        print("\n" + "-"*70)
        print("💡 BETTING INSIGHTS:")
        print("-"*70)
        
        max_prob = probabilities.max()
        outcomes = ['Away Win', 'Draw', 'Home Win']
        predicted_outcome = outcomes[prediction]
        
        if confidence in ['Very High', 'High']:
            print(f"✓ Strong prediction: {predicted_outcome} ({max_prob:.1%})")
            print("  Recommended: Consider betting on this outcome")
        elif confidence == 'Medium':
            print(f"⚠️  Moderate prediction: {predicted_outcome} ({max_prob:.1%})")
            print("  Recommended: Proceed with caution")
        else:
            print(f"⚠️  Low confidence prediction: {predicted_outcome} ({max_prob:.1%})")
            print("  Recommended: Avoid betting - outcome uncertain")
        
        # Show second-most likely outcome
        sorted_probs = sorted(zip(outcomes, probabilities), key=lambda x: x[1], reverse=True)
        second_outcome, second_prob = sorted_probs[1]
        print(f"\n  Alternative: {second_outcome} ({second_prob:.1%})")
        
        # Risk assessment
        if probabilities[1] > 0.35:  # Draw probability high
            print("\n⚠️  HIGH DRAW RISK: Draw probability is significant")
        
        if max_prob < 0.45:
            print("\n⚠️  CLOSE MATCH: All outcomes have similar probabilities")
    
    # Historical head-to-head
    print("\n" + "="*70)
    print("📜 HISTORICAL HEAD-TO-HEAD")
    print("="*70)
    
    h2h_matches = df_processed[
        ((df_processed['HomeTeam'] == home_team) & (df_processed['AwayTeam'] == away_team)) |
        ((df_processed['HomeTeam'] == away_team) & (df_processed['AwayTeam'] == home_team))
    ].tail(5)
    
    if len(h2h_matches) > 0:
        print(f"\nLast {len(h2h_matches)} meetings:\n")
        for idx, match in h2h_matches.iterrows():
            date = match['Date'].strftime('%d/%m/%Y') if pd.notna(match['Date']) else 'N/A'
            score = f"{int(match['FTHG'])}-{int(match['FTAG'])}"
            result_str = match['FTR']
            
            if result_str == 'H':
                result_text = f"🏠 {match['HomeTeam']} won"
            elif result_str == 'A':
                result_text = f"✈️  {match['AwayTeam']} won"
            else:
                result_text = "🤝 Draw"
            
            print(f"  {date}: {match['HomeTeam']} {score} {match['AwayTeam']} - {result_text}")
        
        # H2H summary
        home_wins = len(h2h_matches[(h2h_matches['HomeTeam'] == home_team) & (h2h_matches['FTR'] == 'H')])
        home_wins += len(h2h_matches[(h2h_matches['AwayTeam'] == home_team) & (h2h_matches['FTR'] == 'A')])
        
        away_wins = len(h2h_matches[(h2h_matches['HomeTeam'] == away_team) & (h2h_matches['FTR'] == 'H')])
        away_wins += len(h2h_matches[(h2h_matches['AwayTeam'] == away_team) & (h2h_matches['FTR'] == 'A')])
        
        draws = len(h2h_matches[h2h_matches['FTR'] == 'D'])
        
        print(f"\n  Summary: {home_team} {home_wins}W - {draws}D - {away_wins}W {away_team}")
    else:
        print("\n  No recent head-to-head matches found in dataset")
    
    # Ask for another prediction
    print("\n" + "="*70)
    another = input("\n🔄 Predict another match? (Y/N): ").strip().upper()
    
    if another in ['Y', 'YES']:
        interactive_prediction_enhanced()
    else:
        print("\n" + "="*70)
        print(" "*15 + "Thank you for using EPL Predictor!")
        print(" "*20 + "Good luck with your bets! 🍀")
        print("="*70)

# Start interactive interface
print("\n🚀 Starting Enhanced Interactive Prediction Interface...\n")
interactive_prediction_enhanced()
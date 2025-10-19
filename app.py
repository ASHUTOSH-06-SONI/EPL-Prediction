# app.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- CONFIGURATION ---
MODEL_FILE = 'epl_predictor_model.pkl'
ENCODER_FILE = 'team_encoder.pkl'
DATA_FILE = 'df_processed.pkl'
FEATURES_FILE = 'available_features.pkl'
# ---------------------

@st.cache_resource
def load_artifacts():
    """Loads all necessary trained artifacts with caching."""
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        
        with open(ENCODER_FILE, 'rb') as f:
            team_encoder = pickle.load(f)
            
        df_features = pd.read_pickle(DATA_FILE)
        
        with open(FEATURES_FILE, 'rb') as f:
            available_features = pickle.load(f)
            
        return model, team_encoder, df_features, available_features
    except FileNotFoundError as e:
        st.error(f"FATAL ERROR: Required file not found: {e.filename}. Please run 'train_and_save_model.py' first.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading model artifacts: {e}")
        st.stop()

def predict_match_enhanced(home_team, away_team, model, team_encoder, df_features, available_features):
    """
    Predicts a single match result using the loaded model and feature engineering logic.
    """
    try:
        recent_home_all = df_features[df_features['HomeTeam'] == home_team].tail(1)
        recent_away_all = df_features[df_features['AwayTeam'] == away_team].tail(1)
        
        if len(recent_home_all) == 0 or len(recent_away_all) == 0:
            return None, "Insufficient data for one or both teams."
        
        recent_home = recent_home_all.iloc[0]
        recent_away = recent_away_all.iloc[0]
        
        match_data = {}
        
        match_data['home_team_encoded'] = team_encoder.transform([home_team])[0]
        match_data['away_team_encoded'] = team_encoder.transform([away_team])[0]
        
        for feat in available_features:
            if feat in ['home_team_encoded', 'away_team_encoded']:
                continue
            
            if feat.startswith('home_') or feat.startswith('norm_') or not (feat.startswith('home_') or feat.startswith('away_')):
                if feat in recent_home:
                    match_data[feat] = recent_home[feat]
            elif feat.startswith('away_'):
                if feat in recent_away:
                    match_data[feat] = recent_away[feat]
        
        # Calculate differential features if not present (robustness check)
        if 'goal_diff_5' in available_features and 'goal_diff_5' not in match_data:
            match_data['goal_diff_5'] = match_data.get('home_goals_scored_5', 0) - match_data.get('away_goals_scored_5', 0)
        
        # Ensure all features are present in correct order
        match_features = pd.DataFrame([match_data], columns=available_features)
        match_features = match_features.fillna(0)
        
        prediction = model.predict(match_features)[0]
        probabilities = model.predict_proba(match_features)[0]
        
        outcomes = ['Away Win', 'Draw', 'Home Win']
        max_prob = probabilities.max()
        
        if max_prob > 0.6: confidence = "Very High"
        elif max_prob > 0.5: confidence = "High"
        elif max_prob > 0.4: confidence = "Medium"
        else: confidence = "Low"
        
        return {
            'prediction': outcomes[int(prediction)],
            'probabilities': {outcomes[i]: prob for i, prob in enumerate(probabilities)},
            'confidence': confidence,
            'max_prob': max_prob,
            'recent_home': recent_home,
            'recent_away': recent_away
        }, None
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None, "An internal error occurred during prediction calculation."

# --- MAIN STREAMLIT APP ---
def main():
    st.set_page_config(page_title="EPL Match Predictor", layout="wide")
    st.title("⚽ ENHANCED EPL MATCH PREDICTOR")
    
    model, team_encoder, df_processed, available_features = load_artifacts()
    available_teams = sorted(team_encoder.classes_)
    
    st.markdown("---")

    # --- INPUT SELECTION ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🏠 Home Team")
        home_team = st.selectbox(
            "Select Home Team",
            options=available_teams,
            index=available_teams.index("Arsenal") if "Arsenal" in available_teams else 0
        )
        
    with col2:
        st.header("✈️ Away Team")
        away_options = [team for team in available_teams if team != home_team]
        away_team = st.selectbox(
            "Select Away Team",
            options=away_options,
            index=away_options.index("Manchester City") if "Manchester City" in away_options else 0
        )

    if not home_team or not away_team:
        st.warning("Please select both a Home Team and an Away Team.")
        return

    st.markdown("---")
    
    if st.button("🔮 Predict Match Result", type="primary"):
        
        with st.spinner(f'Calculating prediction for {home_team} vs {away_team}...'):
            
            prediction_result, error = predict_match_enhanced(
                home_team, away_team, model, team_encoder, df_processed, available_features
            )
            
            if error:
                st.error(error)
                return

        # --- PREDICTION RESULT ---
        st.subheader("✅ Match Prediction")
        
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            st.metric(
                label="Predicted Outcome",
                value=prediction_result['prediction'],
                delta=f"Confidence: {prediction_result['confidence']}",
                delta_color="off"
            )
        
        with pred_col2:
            st.metric(
                label="Probability of Prediction",
                value=f"{prediction_result['max_prob']:.1%}"
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Probability Chart
        st.subheader("📊 Outcome Probabilities")
        probs_df = pd.DataFrame(
            {'Probability': prediction_result['probabilities'].values()},
            index=prediction_result['probabilities'].keys()
        ).sort_values(by='Probability', ascending=False)
        
        st.bar_chart(probs_df)

        st.markdown("---")
        
        # --- RECENT FORM ANALYSIS ---
        st.subheader("📊 Recent Form Analysis (Last 5 Games)")
        
        form_col1, form_col2 = st.columns(2)
        
        recent_home = prediction_result['recent_home']
        recent_away = prediction_result['recent_away']
        
        with form_col1:
            st.markdown(f"**🏠 {home_team} (Home)**")
            st.markdown(f"- **Goals Scored**: {recent_home['home_goals_scored_5']:.2f} per game")
            st.markdown(f"- **Goals Conceded**: {recent_home['home_goals_conceded_5']:.2f} per game")
            st.markdown(f"- **Form Points**: {recent_home['home_form_5']:.2f} / 3.00")
            if 'home_win_pct_5' in recent_home:
                st.markdown(f"- **Win Rate**: {recent_home['home_win_pct_5']*100:.1f}%")
        
        with form_col2:
            st.markdown(f"**✈️ {away_team} (Away)**")
            st.markdown(f"- **Goals Scored**: {recent_away['away_goals_scored_5']:.2f} per game")
            st.markdown(f"- **Goals Conceded**: {recent_away['away_goals_conceded_5']:.2f} per game")
            st.markdown(f"- **Form Points**: {recent_away['away_form_5']:.2f} / 3.00")
            if 'away_win_pct_5' in recent_away:
                st.markdown(f"- **Win Rate**: {recent_away['away_win_pct_5']*100:.1f}%")
        
        st.markdown("---")

        # --- H2H AND INSIGHTS ---
        st.subheader("💡 Additional Insights")

        # Historical Head-to-Head (The section that required the fix)
        h2h_matches = df_processed[
            ((df_processed['HomeTeam'] == home_team) & (df_processed['AwayTeam'] == away_team)) |
            ((df_processed['HomeTeam'] == away_team) & (df_processed['AwayTeam'] == home_team))
        ].tail(5)
        
        if len(h2h_matches) > 0:
            st.markdown("##### Historical Head-to-Head (Last 5 Meetings)")
            h2h_summary = []
            for _, match in h2h_matches.iterrows():
                date = match['Date'].strftime('%d/%m/%Y') if pd.notna(match['Date']) else 'N/A'
                # These lines now work because FTHG and FTAG are saved
                score = f"{int(match['FTHG'])}-{int(match['FTAG'])}" 
                
                if match['FTR'] == 'H':
                    result_text = f"{match['HomeTeam']} won"
                elif match['FTR'] == 'A':
                    result_text = f"{match['AwayTeam']} won"
                else:
                    result_text = "Draw"
                
                h2h_summary.append(f"*{date}*: {match['HomeTeam']} **{score}** {match['AwayTeam']} ({result_text})")
            
            st.markdown("\n".join(h2h_summary))
        else:
            st.markdown("No recent head-to-head matches found in dataset.")

        # Betting Insights
        st.markdown("##### Betting/Risk Assessment")
        
        insights = []
        max_prob = prediction_result['max_prob']
        predicted_outcome = prediction_result['prediction']
        
        if prediction_result['confidence'] in ['Very High', 'High']:
            insights.append(f"🟢 **Strong Prediction**: The model has high confidence in a **{predicted_outcome}** outcome.")
        elif prediction_result['confidence'] == 'Medium':
            insights.append(f"🟡 **Moderate Prediction**: Proceed with caution. The model's confidence is moderate.")
        else:
            insights.append(f"🔴 **Low Confidence**: Avoid high-risk betting. The outcome is highly uncertain.")
            
        if prediction_result['probabilities'].get('Draw', 0) > 0.35:
            insights.append("⚠️ **High Draw Risk**: The probability of a Draw is significant (over 35%).")
            
        if max_prob < 0.45:
            insights.append("⚖️ **Close Match**: All three outcomes have similar probabilities; potential for an upset.")

        st.markdown("\n".join(insights))
        
        st.markdown("---")
        st.markdown("Disclaimer: This is a statistical model and is for informational purposes only. Gamble responsibly.")

if __name__ == '__main__':
    main()
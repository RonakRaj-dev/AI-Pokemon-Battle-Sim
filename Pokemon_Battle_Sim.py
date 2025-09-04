import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import json
import os
from time import sleep
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline   
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

type_chart = {
        'Normal': {
            'Rock': 0.5,
            'Ghost': 0,
            'Steel': 0.5
        },
        'Fire': {
            'Fire': 0.5,
            'Water': 0.5,
            'Grass': 2,
            'Ice': 2,
            'Bug': 2,
            'Rock': 0.5,
            'Dragon': 0.5,
            'Steel': 0.5
        },
        'Water': {
            'Fire': 2,
            'Water': 0.5,
            'Grass': 0.5,
            'Ground': 2,
            'Rock': 2,
            'Dragon': 0.5
        },
        'Electric': {
            'Water': 2,
            'Electric': 0.5,
            'Grass': 0.5,
            'Ground': 0,
            'Flying': 2,
            'Dragon': 0.5
        },
        'Grass': {
            'Fire': 0.5,
            'Water': 2,
            'Grass': 0.5,
            'Poison': 0.5,
            'Ground': 2,
            'Flying': 0.5,
            'Bug': 0.5,
            'Rock': 2,
            'Dragon': 0.5,
            'Steel': 0.5
        },
        'Ice': {
            'Fire': 0.5,
            'Water': 0.5,
            'Grass': 2,
            'Ice': 0.5,
            'Ground': 2,
            'Flying': 2,
            'Dragon': 2,
            'Steel': 0.5
        },
        'Fighting': {
            'Normal': 2,
            'Ice': 2,
            'Poison': 0.5,
            'Flying': 0.5,
            'Psychic': 0.5,
            'Bug': 0.5,
            'Rock': 2,
            'Ghost': 0,
            'Dark': 2,
            'Steel': 2,
            'Fairy': 0.5
        },
        'Poison': {
            'Grass': 2,
            'Poison': 0.5,
            'Ground': 0.5,
            'Rock': 0.5,
            'Ghost': 0.5,
            'Steel': 0,
            'Fairy': 2
        },
        'Ground': {
            'Fire': 2,
            'Electric': 2,
            'Grass': 0.5,
            'Poison': 2,
            'Flying': 0,
            'Bug': 0.5,
            'Rock': 2,
            'Steel': 2
        },
        'Flying': {
            'Electric': 0.5,
            'Grass': 2,
            'Fighting': 2,
            'Bug': 2,
            'Rock': 0.5,
            'Steel': 0.5
        },
        'Psychic': {
            'Fighting': 2,
            'Poison': 2,
            'Psychic': 0.5,
            'Dark': 0,
            'Steel': 0.5
        },
        'Bug': {
            'Fire': 0.5,
            'Grass': 2,
            'Fighting': 0.5,
            'Poison': 0.5,
            'Flying': 0.5,
            'Psychic': 2,
            'Ghost': 0.5,
            'Dark': 2,
            'Steel': 0.5,
            'Fairy': 0.5
        },
        'Rock': {
            'Fire': 2,
            'Ice': 2,
            'Fighting': 0.5,
            'Ground': 0.5,
            'Flying': 2,
            'Bug': 2,
            'Steel': 0.5
        },
        'Ghost': {
            'Normal': 0,
            'Psychic': 2,
            'Ghost': 2,
            'Dark': 0.5
        },
        'Dragon': {
            'Dragon': 2,
            'Steel': 0.5,
            'Fairy': 0
        },
        'Dark': {
            'Fighting': 0.5,
            'Psychic': 2,
            'Ghost': 2,
            'Dark': 0.5,
            'Fairy': 0.5
        },
        'Steel': {
            'Fire': 0.5,
            'Water': 0.5,
            'Electric': 0.5,
            'Ice': 2,
            'Rock': 2,
            'Steel': 0.5,
            'Fairy': 2
        },
        'Fairy': {
            'Fire': 0.5,
            'Fighting': 2,
            'Poison': 0.5,
            'Dragon': 2,
            'Dark': 2,
            'Steel': 0.5
        }
    }

# Battle Simulation & Training Data Creation

def simulate_battles(pokemon_df, num_battles=10000):
    """
    Creates battle pairs with synthetic outcomes based on:
    - Type matchups
    - Stat differentials
    - Speed priority
    """
    battle_data = []
    pokemon_list = pokemon_df.to_dict('records')

    for _ in range(num_battles):
        # Randomly select two different Pokémon
        p1, p2 = np.random.choice(pokemon_list, 2, replace=False)

        # Calculate type advantage (handle potential missing Type2)
        type_adv = calculate_net_type_advantage(
            p1['Type1'], p1.get('Type2', None),
            p2['Type1'], p2.get('Type2', None)
        )

        # Calculate stat differential (weighted)
        stat_score = (0.3*(p1['HP'] - p2['HP'])) + (0.4*(p1['Attack'] - p2['Attack'])) + (0.3*(p1['Defence'] - p2['Defence'])) + (0.2*(p1['Sp.Attack']-p2['Sp.Attack'])) + (0.2*(p1['Sp.Defence']-p2['Sp.Defence'])) + (0.1*(p1['Speed']-p2['Speed']))

        # Determine outcome (synthetic label)
        outcome = int(
            (0.6 * type_adv) +
            (0.3 * np.sign(stat_score)) +
            (0.1 * (1 if p1['Speed'] > p2['Speed'] else -1)) > 0
        )

        # Create battle record
        battle_record = {
            'p1_species': p1['Species'],
            'p2_species': p2['Species'],
            'type_advantage': type_adv,
            'stat_differential': stat_score,
            'speed_advantage': p1['Speed'] - p2['Speed'],
            'outcome': outcome
        }

        # Add expanded features
        for stat in ['HP','Attack','Defence','Sp.Attack','Sp.Defence','Speed']:
            battle_record[f'p1_{stat}'] = p1[stat]
            battle_record[f'p2_{stat}'] = p2[stat]

        battle_data.append(battle_record)

    return pd.DataFrame(battle_data)

def calculate_net_type_advantage(p1_type1, p1_type2, p2_type1, p2_type2):
    advantage = 1.0
    atk_types = [t for t in [p1_type1, p1_type2] if pd.notna(t) and t]
    def_types = [t for t in [p2_type1, p2_type2] if pd.notna(t) and t]

    for atk in atk_types:
        for dfn in def_types:
            advantage *= type_chart.get(atk, {}).get(dfn, 1)

    if advantage == 0:
        return -4
    elif advantage == 1:
        return 0
    else:
        return np.clip(np.log2(advantage), -4, 4)
        

# ----------- Feature Engineering Pipeline ------------

def create_feature_pipeline():
    # Numerical features to scale
    numeric_features = [
        'type_advantage',
        'stat_differential',
        'speed_advantage',
        'p1_HP', 'p1_Attack', 'p1_Defence',
        'p2_HP', 'p2_Attack', 'p2_Defence'
    ]

    # Categorical features to encode
    categorical_features = ['p1_species', 'p2_species']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    return preprocessor


# ----------------- Model Training with Synthetic Data -----------------

def train_battle_predictor(pokemon_df, n_estimators=200, max_depth=10, test_size=0.2, random_state=42):
    """
    Enhanced battle predictor training function with additional features:
    - Detailed evaluation metrics
    - Feature importance visualization
    - Flexible hyperparameters
    - Better random state control
    """
    # Step 1: Generate battle data
    battle_df = simulate_battles(pokemon_df)

    # Step 2: Prepare features and target
    X = battle_df.drop('outcome', axis=1)
    y = battle_df['outcome']

    # Step 3: Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class balance
    )

    # Step 4: Create pipeline
    pipeline = Pipeline([
        ('preprocessor', create_feature_pipeline()),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1  # Use all available cores
        ))
    ])

    # Step 5: Train model with timing
    import time
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Step 6: Enhanced evaluation
    print("\n=== Model Evaluation ===")

    # Training and test accuracy
    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    print(f"\nAccuracy:")
    print(f"- Training: {train_acc:.2%}")
    print(f"- Test:     {test_acc:.2%}")

    # Detailed classification report
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Loss', 'Win'],
                yticklabels=['Loss', 'Win'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Feature importance (if available)
    try:
        feature_importances = pipeline.named_steps['classifier'].feature_importances_
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        plt.title('Top 15 Feature Importances')
        plt.show()
    except Exception as e:
        print(f"\nCould not generate feature importances: {e}")

    # Training time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    return pipeline


# ------------ Prediction Function for New Battles ------------

import difflib
def predict_battle_outcome(model, pokemon_df, pokemon1_name, pokemon2_name):
    df = pokemon_df.copy()
    df['Species'] = df['Species'].astype(str).str.strip().str.lower()
    p1_name, p2_name = pokemon1_name.strip().lower(), pokemon2_name.strip().lower()

    for name in [p1_name, p2_name]:
        if name not in df['Species'].values:
            closest = difflib.get_close_matches(name, df['Species'], n=1)
            raise ValueError(f"{name} not found. Did you mean {closest[0] if closest else 'None'}?")

    p1 = df[df['Species'] == p1_name].iloc[0]
    p2 = df[df['Species'] == p2_name].iloc[0]

    record = {
        'p1_species': p1['Species'],
        'p2_species': p2['Species'],
        'type_advantage': calculate_net_type_advantage(p1['Type1'], p1.get('Type2', 'None'),
                                                       p2['Type1'], p2.get('Type2', 'None')),
        'stat_differential': (
            0.3*(p1['HP'] - p2['HP']) +
            0.4*(p1['Attack'] - p2['Attack']) +
            0.3*(p1['Defence'] - p2['Defence'])
        ),
        'speed_advantage': p1['Speed'] - p2['Speed']
    }

    for stat in ['HP', 'Attack', 'Defence', 'Sp.Attack', 'Sp.Defence', 'Speed']:
        record[f'p1_{stat}'] = p1[stat]
        record[f'p2_{stat}'] = p2[stat]

    battle_df = pd.DataFrame([record])
    proba = model.predict_proba(battle_df)[0]
    prediction = model.predict(battle_df)[0]

    return {
        'probability_p1_wins': proba[1],
        'predicted_winner': p1['Species'] if prediction == 1 else p2['Species']
    }

    
    
# --------------- Pokemon Expected Damage Output ---------------
    
def calculate_damage(attacker, defender, move_power, move_type, is_physical=True):
    """
    Calculate expected damage from attacker to defender
    
    Parameters:
    - attacker: Dict containing attacker stats
    - defender: Dict containing defender stats
    - move_power: Base power of the move
    - move_type: Type of the move
    - is_physical: Whether move is physical (True) or special (False)
    
    Returns:
    - Expected damage value
    """
    # Determine attacking and defending stats
    attack_stat = attacker['Attack'] if is_physical else attacker['Sp.Attack']
    defence_stat = defender['Defence'] if is_physical else defender['Sp.Defence']
    
    # Get attacker level (default to 50 if not available)
    level = attacker.get('level', 50)
    
    # Base damage calculation
    base_damage = (((2 * level / 5 + 2) * move_power * attack_stat / defence_stat) / 50) + 2
    
    # Calculate modifiers
    modifiers = 1.0
    
    # STAB (Same Type Attack Bonus)
    if move_type in [attacker['Type1'], attacker.get('Type2', None)]:
        modifiers *= 1.5
    
    # Type effectiveness
    type_effectiveness = calculate_type_effectiveness(
        move_type, 
        defender['Type1'], 
        defender.get('Type2', None)
    )
    modifiers *= type_effectiveness
    
    # Random factor (average)
    modifiers *= 0.925  # Average of 0.85-1.00
    
    # Calculate final damage
    damage = base_damage * modifiers
    
    return max(1, int(round(damage)))  # Minimum damage is 1

def calculate_type_effectiveness(move_type, def_type1, def_type2=None):
    """
    Calculate type effectiveness multiplier
    """
    
    effectiveness = 1.0
    
    # Check against first type
    effectiveness *= type_chart.get(move_type, {}).get(def_type1, 1)
    
    # Check against second type if exists
    if def_type2 and pd.notna(def_type2):
        effectiveness *= type_chart.get(move_type, {}).get(def_type2, 1)
    
    return effectiveness


# CLean and Load the Data for any anomalies in the data set

def load_and_clean_data(filepath):
    pokemon_data = pd.read_csv(filepath, encoding='ISO-8859-1')

    # Rename incorrect columns
    if 'Tyep2' in pokemon_data.columns:
        pokemon_data.rename(columns={'Tyep2': 'Type2'}, inplace=True)

    # Fill missing categorical values
    fill_values = {
        'Type2': 'None',
        'Ability1': 'Unknown', 'Ability2': 'Unknown', 'Ability3': 'Unknown',
        'Egg Type1': 'Unknown', 'Egg Type2': 'Unknown', 'Moves': 'Unknown'
    }
    pokemon_data.fillna(value=fill_values, inplace=True)

    # Fill missing numerics with default values
    numeric_cols = ['HP', 'Attack', 'Defence', 'Sp.Attack', 'Sp.Defence', 'Speed', 'level', 'Weight', 'Height']
    for col in numeric_cols:
        if col in pokemon_data.columns:
            pokemon_data[col] = pokemon_data[col].fillna(0)

    # Standardize Species column
    if 'Species' in pokemon_data.columns:
        pokemon_data['Species'] = pokemon_data['Species'].astype(str).str.strip().str.lower()

    return pokemon_data

# --------------- Rank Pokemon by battle effectiveness code ---------------


def rank_pokemon_by_effectiveness(pokemon_df, num_simulations=1000):
    """
    Rank Pokémon by battle effectiveness through simulated matchups
    
    Args:
        pokemon_df: DataFrame containing Pokémon data
        num_simulations: Number of matchups to simulate per Pokémon
        
    Returns:
        DataFrame with Pokémon ranked by battle effectiveness
    """
    # Preprocessing
    pokemon_df = pokemon_df.copy()
    pokemon_df.fillna({'Type2': 'None'}, inplace=True)
    
    # Calculate stat totals with weighting
    stat_weights = {
        'HP': 0.8,
        'Attack': 1.0,
        'Defence': 0.9,
        'Sp.Attack': 1.0,
        'Sp.Defence': 0.9,
        'Speed': 1.2  # Speed is more valuable
    }
    
    for stat, weight in stat_weights.items():
        pokemon_df[f'weighted_{stat}'] = pokemon_df[stat] * weight
    
    # Normalize stats
    scaler = StandardScaler()
    stat_cols = [f'weighted_{stat}' for stat in stat_weights]
    pokemon_df[stat_cols] = scaler.fit_transform(pokemon_df[stat_cols])
    
    # Calculate overall stat score
    pokemon_df['stat_score'] = pokemon_df[stat_cols].mean(axis=1)
    
    # Get list of all Pokémon records
    all_pokemon = pokemon_df.to_dict('records')
    num_pokemon = len(all_pokemon)
    
    # Adjust simulations if needed
    if num_simulations > num_pokemon - 1:
        print(f"Reducing simulations from {num_simulations} to {num_pokemon - 1} (dataset size)")
        num_simulations = num_pokemon - 1
    
    def evaluate_pokemon(pokemon):
        # Sample opponents without replacement, then with replacement if needed
        opponents = []
        remaining_pokemon = [p for p in all_pokemon if p['Species'] != pokemon['Species']]
        
        if len(remaining_pokemon) >= num_simulations:
            opponents = np.random.choice(remaining_pokemon, num_simulations, replace=False)
        else:
            # If we don't have enough unique opponents, sample with replacement
            opponents = np.random.choice(remaining_pokemon, num_simulations, replace=True)
        
        wins = sum(simulate_matchup(pokemon, opp) for opp in opponents)
        return wins / num_simulations
    
    def simulate_matchup(pokemon, opponent):
        type_advantage = calculate_net_type_advantage(
            pokemon['Type1'], pokemon['Type2'],
            opponent['Type1'], opponent['Type2']
        )
        stat_difference = pokemon['stat_score'] - opponent['stat_score']
        return 1 if (0.6 * type_advantage + 0.4 * stat_difference) > 0 else 0
    
    # Evaluate each Pokémon
    pokemon_df['win_rate'] = pokemon_df.apply(
        lambda row: evaluate_pokemon(row.to_dict()), 
        axis=1
    )
    
    # Composite score combining stats and win rate
    pokemon_df['effectiveness_score'] = (
        0.7 * pokemon_df['win_rate'] + 
        0.3 * pokemon_df['stat_score']
    )
    
    # Rank Pokémon
    ranked_pokemon = pokemon_df.sort_values(
        'effectiveness_score', 
        ascending=False
    )[['Species', 'Type1', 'Type2', 'effectiveness_score', 'win_rate'] + stat_cols]
    
    ranked_pokemon['rank'] = range(1, len(ranked_pokemon) + 1)
    
    return ranked_pokemon



# ----------------- Group Pokemons by Battle Roles -----------------

# Configuration
MOVES_COLUMN = 'Moves'
POKEMON_NAME_COL = 'Species'
API_CACHE_FILE = 'move_cache.json'
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
MAX_WORKERS = 5  # Reduced to avoid rate limiting

# Load or initialize move cache
if os.path.exists(API_CACHE_FILE):
    with open(API_CACHE_FILE, 'r') as f:
        move_cache = json.load(f)
else:
    move_cache = {}

def save_cache():
    with open(API_CACHE_FILE, 'w') as f:
        json.dump(move_cache, f)

def verify_move_with_api(move_name):
    """Check if a move exists in PokeAPI with retry logic"""
    standardized_name = move_name.lower().replace(' ', '-')
    
    if standardized_name in move_cache:
        return move_cache[standardized_name]
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                f"https://pokeapi.co/api/v2/move/{standardized_name}",
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                data = response.json()
                result = {
                    'exists': True,
                    'name': data['name'],
                    'type': data['type']['name'],
                    'damage_class': data['damage_class']['name'],
                    'power': data['power'],
                    'priority': data['priority']
                }
                move_cache[standardized_name] = result
                return result
            elif response.status_code == 404:
                result = {'exists': False, 'name': move_name}
                move_cache[standardized_name] = result
                return result
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"Failed to fetch {move_name}: {str(e)}")
                result = {'exists': False, 'name': move_name}
                move_cache[standardized_name] = result
                return result
            sleep(1)  # Wait before retrying
    
    return {'exists': False, 'name': move_name}

def classify_roles_with_hybrid_data(pokemon_df):
    """Optimized role classification with better progress tracking"""
    
    # 1. Extract and verify moves
    def extract_moves(move_str):
        if pd.isna(move_str):
            return []
        return [move.strip() for move in str(move_str).split(',')]
    
    pokemon_df['parsed_moves'] = pokemon_df[MOVES_COLUMN].apply(extract_moves)
    all_moves = list({move for moves in pokemon_df['parsed_moves'] for move in moves})
    
    print(f"Verifying {len(all_moves)} unique moves...")
    
    # Verify moves with better progress tracking
    verified_moves = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for move in all_moves:
            futures.append(executor.submit(verify_move_with_api, move))
        
        for future in tqdm(futures, desc="Processing moves"):
            result = future.result()
            verified_moves[result['name']] = result
    
    # 2. Classify moves
    move_categories = {}
    for move_name, move_data in verified_moves.items():
        if not move_data['exists']:
            continue
            
        categories = []
        if move_data['power']:
            if move_data['damage_class'] == 'physical':
                categories.append('physical_attack')
            elif move_data['damage_class'] == 'special':
                categories.append('special_attack')
        else:
            if any(x in move_name for x in ['swords-dance', 'dragon-dance', 'bulk-up']):
                categories.append('physical_setup')
            elif any(x in move_name for x in ['nasty-plot', 'calm-mind', 'quiver-dance']):
                categories.append('special_setup')
            elif any(x in move_name for x in ['stealth-rock', 'spikes', 'toxic-spikes']):
                categories.append('hazard')
            elif any(x in move_name for x in ['u-turn', 'volt-switch', 'flip-turn']):
                categories.append('pivot')
        
        if categories:
            move_categories[move_name] = categories
    
    # 3. Determine roles
    def determine_role(row):
        features = {
            'physical_attack': False,
            'special_attack': False,
            'physical_setup': False,
            'special_setup': False,
            'hazard': False,
            'pivot': False
        }
        
        for move in row['parsed_moves']:
            api_name = move.lower().replace(' ', '-')
            if api_name in move_categories:
                for category in move_categories[api_name]:
                    if category in features:
                        features[category] = True
        
        if features['hazard']:
            return 'Hazard Setter'
        elif features['pivot']:
            return 'Pivot'
        elif features['physical_setup'] and features['physical_attack']:
            return 'Physical Setup Sweeper'
        elif features['special_setup'] and features['special_attack']:
            return 'Special Setup Sweeper'
        elif features['physical_attack']:
            return 'Physical Attacker'
        elif features['special_attack']:
            return 'Special Attacker'
        return 'Balanced'
    
    print("Classifying Pokémon roles...")
    pokemon_df['role'] = pokemon_df.progress_apply(determine_role, axis=1)
    
    # 4. Generate reports
    move_report = [
        {
            'move': name,
            'exists_in_api': data['exists'],
            'type': data.get('type'),
            'damage_class': data.get('damage_class'),
            'power': data.get('power')
        }
        for name, data in verified_moves.items()
    ]
    
    save_cache()
    return pokemon_df, pd.DataFrame(move_report)


# -------------- Predict Tournament Bracket Outcomes ----------------

class TournamentPredictor:
    def __init__(self, pokemon_data):
        self.pokemon = pokemon_data
        self.moves = pokemon_data
        self.le = LabelEncoder()
        self.model = None
        self.type_chart = self.create_type_chart()
        self.pokemon.columns = self.pokemon.columns.str.strip()
    
        # Verify Species column exists
        if 'Species' not in self.pokemon.columns:
            raise ValueError("Data must contain 'Species' column")
            
        # Clean Species data
        self.pokemon['Species'] = self.pokemon['Species'].astype(str).str.strip()
        if self.pokemon['Species'].isnull().any():
            raise ValueError("Species column contains null values")
            
        
    def create_type_chart(self):
        return type_chart
    
    def calculate_matchup(self, p1, p2):
        """ Calculate matchup score between two Pokémon """
        effectiveness = 1.0
        for p1_type in [p1['Type1'], p1['Type2'] if 'Type2' in p1 and pd.notna(p1['Type2']) else None]:
            if pd.isna(p1_type): continue
            for p2_type in [p2['Type1'], p2['Type2'] if 'Type2' in p2 and pd.notna(p2['Type2']) else None]:
                if pd.isna(p2_type): continue
                effectiveness *= self.type_chart.get(p1_type, {}).get(p2_type, 1)

        return effectiveness
    
    def simulate_battle(self, p1, p2, n_simulations=100):
        """ Simulate multiple battles between two Pokémon """
        p1_wins = 0
        for _ in range(n_simulations):
            if self.simulate_single_battle(p1, p2):
                p1_wins += 1
        return p1_wins / n_simulations
    
    def simulate_single_battle(self, p1, p2):
        """ Single Battle Simulation with move selection"""
        
        p1_dmg = self.calculate_damage(p1, p2)
        p2_dmg = self.calculate_damage(p2, p1)
        
        p1_first = p1['Speed'] > p2['Speed']
        
        if p1_first and p1_dmg >= p2['HP']:
            return True
        elif not p1_first and p2_dmg >= p1['HP']:
            return False
        return p1_dmg > p2_dmg
    
    def calculate_damage(self, attacker, defender):
        """ Calculate expected damage from attacker to defender"""
        
        best_damage = 0
        for move in self.get_moves(attacker['Species']):
            effectiveness = 1
            if move['type'] in [defender['Type1'], defender['Type2'] if 'Type2' in defender and pd.notna(defender['Type2']) else None]:
                effectiveness = self.type_chart.get(move['type'], {}).get(defender['Type1'], 1)

            stab = 1.5 if move['type'] in [attacker['Type1'], attacker['Type2'] if 'Type2' in attacker and pd.notna(attacker['Type2']) else None] else 1

            damage = (move['power'] * (attacker['Attack'] if move['category'] == 'Physical' else attacker['Sp.Attack']) / (defender['Defence'] if move['category'] == 'Physical' else defender['Sp.Defence']) * effectiveness * stab)
            
            if damage > best_damage:
                best_damage = damage
        return best_damage
    
    def get_moves(self, species):
        """ Get moves for a pokemon species """
        pokemon_row = self.moves[self.moves['Species'] == species]
        if pokemon_row.empty:
            return []
        
        # Parse the moves string and return as list of dictionaries
        moves_str = pokemon_row['Moves'].iloc[0]
        if pd.isna(moves_str):
            return []
        
        # Return basic move structure - you may need to adjust based on your move data format
        moves_list = [move.strip() for move in str(moves_str).split(',')]
        return [{'type': 'Normal', 'power': 50, 'category': 'Physical'} for _ in moves_list]  # Placeholder values
    
    def train_model(self):
        """Train prediction model on historical battle data"""
        try:
            # First verify the Species column exists and is valid
            if 'Species' not in self.pokemon.columns:
                raise ValueError("'Species' column not found in Pokémon data")
                
            # Clean the Species column
            self.pokemon['Species'] = self.pokemon['Species'].astype(str).str.strip()
            if self.pokemon['Species'].isnull().any():
                self.pokemon['Species'] = self.pokemon['Species'].fillna('Unknown')
            
            features = []
            targets = []
            
            # Get unique species to avoid duplicate comparisons
            unique_species = self.pokemon['Species'].unique()
            
            # Progress bar for training
            with tqdm(total=len(unique_species)*(len(unique_species)-1)//2, 
                    desc="Training model") as pbar:
                
                for i, p1_species in enumerate(unique_species.tolist()):
                    p1_match = self.pokemon.loc[self.pokemon['Species'] == str(p1_species)]
                    if p1_match.empty:
                        continue
                    p1 = p1_match.iloc[0]
                    
                    # Only compare with remaining species to avoid duplicates
                    for p2_species in unique_species[i+1:]:
                        p2_match = self.pokemon.loc[self.pokemon['Species'] == str(p2_species)]
                        if p2_match.empty:
                            continue
                        p2 = p2_match.iloc[0]
                        
                        type_advantage = self.calculate_matchup(p1, p2)
                        p1_dict = p1.to_dict()
                        p2_dict = p2.to_dict()
                        stat_diff = np.array([p1_dict[col] - p2_dict[col] for col in ['HP', 'Attack', 'Defence', 'Sp.Attack', 'Sp.Defence', 'Speed']])
                        
                        features.append(np.concatenate([np.array([type_advantage]), stat_diff]))
                        targets.append(self.simulate_battle(p1, p2) > 0.5)
                        pbar.update(1)
            
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(features, targets)
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            # Print debug information
            print("\nDebug Info:")
            print(f"Columns in data: {self.pokemon.columns.tolist()}")
            print(f"Species sample: {self.pokemon['Species'].head() if 'Species' in self.pokemon.columns else 'N/A'}")
            raise
        
    def predict_bracket(self, p1_species, p2_species):
        """Predict outcomes of a single matchup in a tournament bracket"""
        try:
            # Standardize input names
            p1_species = str(p1_species).strip().title()
            p2_species = str(p2_species).strip().title()
            
            # Verify Pokémon exist
            valid_species = self.pokemon['Species'].unique()
            if p1_species not in valid_species:
                closest = difflib.get_close_matches(p1_species, valid_species, n=1)
                raise ValueError(f"'{p1_species}' not found. Did you mean: {closest[0] if closest else 'None'}")
            if p2_species not in valid_species:
                closest = difflib.get_close_matches(p2_species, valid_species, n=1)
                raise ValueError(f"'{p2_species}' not found. Did you mean: {closest[0] if closest else 'None'}")

            # Get Pokémon data
            p1 = self.pokemon[self.pokemon['Species'] == p1_species].iloc[0]
            p2 = self.pokemon[self.pokemon['Species'] == p2_species].iloc[0]
            
            # Calculate features
            type_advantage = self.calculate_matchup(p1, p2)
            stat_diff = (p1[['HP', 'Attack', 'Defence', 'Sp.Attack', 'Sp.Defence', 'Speed']] - 
                        p2[['HP', 'Attack', 'Defence', 'Sp.Attack', 'Sp.Defence', 'Speed']])
            
            features = np.concatenate([
                [type_advantage],
                stat_diff.values
            ]).reshape(1, -1)
            
            # Make prediction
            proba = self.model.predict_proba(features)[0][1]
            return p1_species if proba > 0.5 else p2_species, proba
            
        except Exception as e:
            print(f"Error predicting bracket: {str(e)}")
            raise
        
    

# ----------------------- Main Execution Block ----------------

try:
    import random
    from tqdm import tqdm
    import joblib
    from datetime import datetime
    tqdm.pandas()
    
    # Load and clean data
    pokemon_data = load_and_clean_data('pokemon_final.csv')
    print(pokemon_data.columns.tolist())
    # print('Type2' in pokemon_data.columns)  # Should print True
    # pokemon_data['Species'] = pokemon_data['Species'].fillna("Unknown_Pokemon")
    # print(pokemon_data['Species'].isna().sum())  # Should now output 0
    # print("Columns after cleaning:", pokemon_data.columns.tolist())
    # print([col for col in pokemon_data.columns if 'species' in col.lower()])  # Check for variations |||
    
    pokemon_data['Species'] = pokemon_data['Species'].astype(str).str.strip()
    if pokemon_data['Species'].isnull().any():
        pokemon_data['Species'] = pokemon_data['Species'].fillna('Unknown')

    # Verify other required columns exist
    required_columns = ['Index', 'Species', 'Type1', 'Type2', 'Ability1', 'Ability2', 'Ability3', 'level', 'base_experience', 'HP', 'Attack', 'Defence', 'Sp.Attack', 'Sp.Defence', 'Speed', 'Total', 'Weight', 'Height', 'Seed Type', 'Egg Type1', 'Egg Type2', 'Moves']
    for col in required_columns:
        if col not in pokemon_data.columns:
            raise ValueError(f"Missing required column: {col}")

    print("\nMissing values after cleaning:")
    print(pokemon_data.isna().sum())
    
    # --- Battle Simulation ---
    battle_model = train_battle_predictor(pokemon_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"battle_prediction_model_{timestamp}.pkl"
    joblib.dump(battle_model, model_filename)
    joblib.dump(battle_model, "latest_battle_model.pkl")
    print(f"✅ Battle prediction model saved as: {model_filename}")
    print("✅ Latest model saved as: latest_battle_model.pkl")
    def get_random_pokemon(df):
        return df.sample(1)['Species'].values[0]   
    pokemon1 = get_random_pokemon(pokemon_data)
    pokemon2 = get_random_pokemon(pokemon_data)
    result = predict_battle_outcome(
        battle_model,
        pokemon_data,
        pokemon1,
        pokemon2
    )
    
    # --- Expected Damage Dealt ---
    damage_of_pokemon1 = calculate_damage(
        pokemon_data[pokemon_data['Species'] == pokemon1].iloc[0],
        pokemon_data[pokemon_data['Species'] == pokemon2].iloc[0],
        pokemon_data[pokemon_data['Species'] == pokemon1].iloc[0]['Attack'],  # Example move power
        pokemon_data[pokemon_data['Species'] == pokemon1].iloc[0]['Type1'],  # Example move type
        is_physical=True  # Assuming physical move for this example
    )
    damage_of_pokemon2 = calculate_damage(
        pokemon_data[pokemon_data['Species'] == pokemon1].iloc[0],
        pokemon_data[pokemon_data['Species'] == pokemon2].iloc[0],
        pokemon_data[pokemon_data['Species'] == pokemon2].iloc[0]['Attack'],  # Example move power
        pokemon_data[pokemon_data['Species'] == pokemon2].iloc[0]['Type1'],  # Example move type
        is_physical=True  # Assuming physical move for this example
    )
    
    # --- Rank Pokemob by damage dealt ---
    ranked_pokemon = rank_pokemon_by_effectiveness(pokemon_data, num_simulations=500)
    
    # --- Group pokemon by battle roles ---
    classified_pokemon, move_report = classify_roles_with_hybrid_data(pokemon_data)
    classified_pokemon.to_excel('pokemon_with_roles.xlsx', index=False)
    move_report.to_excel('move_verification.xlsx', index=False)
    
    # --- Pokemon tournament bracket prediction ---
    species_list = pokemon_data['Species'].tolist()
    random.shuffle(species_list)
    predictor = TournamentPredictor(pokemon_data)
    print("\nData Validation Check:")
    print("Columns:", predictor.pokemon.columns.tolist())
    print("Species sample:", predictor.pokemon['Species'].head())
    print("Null values in Species:", predictor.pokemon['Species'].isnull().sum())
    predictor.train_model()
    predictor_filename = f"tournament_predictor_{timestamp}.pkl"
    joblib.dump(predictor, predictor_filename)
    joblib.dump(predictor, "latest_tournament_predictor.pkl")
    print(f"✅ Tournament predictor saved as: {predictor_filename}")
    print("✅ Latest tournament predictor saved as: latest_tournament_predictor.pkl")
    tournament_bracket = list(zip(species_list[::2], species_list[1::2]))
    results = []
    for p1, p2 in tournament_bracket:
        winner, confidence = predictor.predict_bracket(p1, p2)
        results.append((winner, confidence))
    
    
    # --- ALl the print statement for each method above ---
    print(f"\nBattle between {pokemon1} vs {pokemon2}")
    print(f"Predicted winner: {result['predicted_winner']}")
    print(f"Win probability: {result['probability_p1_wins']:.2%}")
    print(f"Expected damage from {pokemon1} to {pokemon2}: {damage_of_pokemon1}")
    print(f"Expected damage from {pokemon2} to {pokemon1}: {damage_of_pokemon2}")
    print(ranked_pokemon.head(20).to_string(index=False))
    
    print("\nProcessing complete!")
    print(f"Found {len(move_report[move_report['exists_in_api']])} valid moves")
    print(f"Found {len(move_report[~move_report['exists_in_api']])} invalid moves")
    
    print("\nTournament Predictions:")
    for (p1, p2), (winner, confidence) in zip(tournament_bracket, results):
        print(f"{p1} vs {p2}: {winner} wins ({confidence:.1%} confidence)")

except Exception as e:
    print(f"\nError occurred: {str(e)} at line: {e.__traceback__.tb_lineno}")
    print(f"❌ Error saving battle model: {str(e)}")
    print(f"❌ Error saving tournament predictor: {str(e)}")
    # Additional debug information
    if 'pokemon_data' in locals():
        print("\nSample data for debugging:")
        print(pokemon_data.head())
        print("\nData types:")
        print(pokemon_data.dtypes)
    # print(f"An error occurred: {str(e)}")
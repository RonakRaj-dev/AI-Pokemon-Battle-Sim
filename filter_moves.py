import pandas as pd
import requests
import time

# Cache to store valid moves per Pokémon species
move_cache = {}

def get_valid_moves(species):
    """Fetch valid moves for a given Pokémon from the PokeAPI, with caching."""
    species_lower = species.lower()

    # Return cached result if available
    if species_lower in move_cache:
        return move_cache[species_lower]

    url = f"https://pokeapi.co/api/v2/pokemon/{species_lower}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"❌ Failed to fetch data for {species} (Status: {response.status_code})")
            move_cache[species_lower] = set()
            return set()

        data = response.json()
        valid_moves = {move['move']['name'] for move in data['moves']}
        move_cache[species_lower] = valid_moves
        return valid_moves

    except requests.RequestException as e:
        print(f"⚠️ Request error for {species}: {e}")
        return set()

def clean_pokemon_moves_csv(csv_file_path, output_file_path=None):
    """Clean the Moves column in a Pokémon CSV file by removing invalid moves."""
    try:
        df = pd.read_csv(csv_file_path, encoding='latin1')
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    if 'Species' not in df.columns or 'Moves' not in df.columns:
        print("❌ CSV must contain 'Species' and 'Moves' columns.")
        return

    filtered_moves = []

    for index, row in df.iterrows():
        species = str(row['Species']).strip()
        raw_moves = str(row['Moves']).lower().replace(" ", "").split(',')

        print(f"🔍 Processing {species} with moves: {raw_moves}")

        valid_moves = get_valid_moves(species)
        filtered = [move for move in raw_moves if move in valid_moves]
        filtered_moves.append(', '.join(filtered))

        # # Sleep to respect API rate limits
        # time.sleep(1)

    df['filtered_moves'] = filtered_moves

    output_path = output_file_path if output_file_path else csv_file_path
    try:
        df.to_csv(output_path, index=False)
        print(f"\n✅ Filtered data saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving output CSV: {e}")

# === USAGE ===
if __name__ == "__main__":
    input_csv = 'pokemon_final.csv'                # Input CSV path
    output_csv = 'filtered_pokemon_moves.csv'      # Output CSV path
    clean_pokemon_moves_csv(input_csv, output_csv)

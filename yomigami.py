import json
import MeCab
import math
import sys

MODEL_FILE = 'yomigami_model.json'

def kata_to_hira(katakana_string):
    """Convert a katakana string to hiragana."""
    hiragana = []
    for char in katakana_string:
        code = ord(char)
        if 12449 <= code <= 12534: 
            hiragana.append(chr(code - 96)) 
        else:
            hiragana.append(char)
    return "".join(hiragana)

def viterbi(model, observations, mecab_readings): 
    """
    Run the Viterbi algorithm.
    Returns (best_path, cumulative log-prob sequence).
    """
    
    T = len(observations)
    if T == 0:
        return ([], [])  # empty path and prob sequence

    # unpack parameters from the model
    start_p = model['priors']
    trans_p = model['transitions']
    emit_p = model['emissions']

    V = [{}]  # V[t][y] holds max log prob at time t for state y
    path = {}
    
    # cumulative log-prob sequence for the best path
    final_delta_sequence = [] 
    
    start_obs = observations[0]
    start_hint = mecab_readings[0] 
    
    possible_states_t0 = emit_p.get(start_obs, {}).keys()
    
    if not possible_states_t0:
        if start_hint != '*':
            possible_states_t0 = [start_hint] 
        else:
            # fallback to the surface form if we have no hint and no emission entry
            possible_states_t0 = [start_obs]

    # 1. Initialization (t=0)
    for y in possible_states_t0:
        emission_prob = emit_p.get(start_obs, {}).get(y, 1e-10) 
        prior_prob = start_p.get(y, start_p.get('__DEFAULT__', 1e-10))
        
        V[0][y] = math.log(prior_prob) + math.log(emission_prob)
        path[y] = [y]

    # 2. Recursion
    for t in range(1, T):
        V.append({})
        new_path = {}
        
        obs = observations[t]
        hint = mecab_readings[t] 
        
        possible_states_t = emit_p.get(obs, {}).keys()
        
        if not possible_states_t:
            if hint != '*':
                possible_states_t = [hint]
            else:
                possible_states_t = [obs]
        
        for y in possible_states_t:
            emission_prob = emit_p.get(obs, {}).get(y, 1e-10) 
            
            # find the best previous state transition
            # (prob, state) = max( V[t-1][y_prev] + log(P(y | y_prev)) )
            (max_prob, best_prev_state) = max(
                (V[t-1][y_prev] + math.log(trans_p.get(y_prev, {}).get(y, trans_p.get('__DEFAULT__', 1e-10))) + math.log(emission_prob), y_prev)
                for y_prev in V[t-1] 
            )
            
            V[t][y] = max_prob
            # store backpointer only; path rebuilt at the end
            new_path[y] = best_prev_state 
            
        # path stores backpointers, not the full path
        path[t] = new_path
        
    # 3. Termination and backtrace
    if not V[T-1]:
        print("  > Warning: unable to decode sentence.")
        return (["?"] * T, [-math.inf] * T)

    # find (best final prob, best last state)
    (max_prob, best_last_state) = max((V[T-1][y], y) for y in V[T-1])
    
    # reconstruct best path and log prob sequence
    best_path_sequence = [best_last_state]
    log_probs_sequence = [max_prob]
    
    # backtrace
    current_state = best_last_state
    for t in range(T - 2, -1, -1):
        current_state = path[t+1][current_state]
        best_path_sequence.insert(0, current_state)
        # record cumulative log prob
        log_probs_sequence.insert(0, V[t][current_state]) 


    # return (best path, cumulative log prob sequence)
    return (best_path_sequence, log_probs_sequence)


def main():
    print("Loading model...")
    try:
        with open(MODEL_FILE, 'r', encoding='utf-8') as f:
            model = json.load(f)
    except FileNotFoundError:
        print(f"Error: model file not found '{MODEL_FILE}'")
        sys.exit(1)
        
    try:
        tagger = MeCab.Tagger("-r /opt/homebrew/etc/mecabrc -d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd")
    except RuntimeError as e:
        print(f"MeCab initialization failed: {e}")
        sys.exit(1)
        
    print("\n--- YomiGami (v2.1 with probabilities) ---")
    print("Enter a Japanese sentence without readings (e.g., watashi wa gakusei desu)")
    print("Type 'q' to quit.")

    while True:
        try:
            sentence = input("> ")
            if sentence.lower() == 'q':
                print("Goodbye!")
                break
            
            node = tagger.parseToNode(sentence)
            
            observations = []  # kanji sequence
            original_words = []  # original words (same as observations, kept for clarity)
            mecab_readings = []  # MeCab reading hints
            
            while node:
                surface = node.surface
                if surface and surface != 'EOS': 
                    observations.append(surface)
                    original_words.append(surface)
                    
                    features = node.feature.split(',')
                    reading_kata = features[7] if len(features) > 7 else '*'
                    reading_hira = kata_to_hira(reading_kata)
                    mecab_readings.append(reading_hira)
                    
                node = node.next
            
            best_path, log_probs_sequence = viterbi(model, observations, mecab_readings)
            
            # 3. Render output with cumulative log probs
            output_parts = []
            
            for i in range(len(observations)):
                word = observations[i]
                reading = best_path[i]
                word_log_prob = log_probs_sequence[i] 
                
                output_parts.append(f"{word}[{reading}] (LogProb: {word_log_prob:.4f})")
            
            print(" > ", " ".join(output_parts))
            
            final_log_prob = log_probs_sequence[-1]
            print(f" > (Total LogProb: {final_log_prob:.4f})")

        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == '__main__':
    main()

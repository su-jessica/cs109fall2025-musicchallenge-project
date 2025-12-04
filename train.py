import json
from collections import defaultdict, Counter

CORPUS_FILE = 'corpus_raw.json'
MODEL_FILE = 'yomigami_model.json'
Aozora_FILE = 'corpus_raw.json'
CUSTOM_PROCESSED_FILE = 'corpus_custom.json' # <-- 添加这个新常量

# 拉普拉斯平滑 (加一平滑)
LAPLACE_SMOOTHING = 1.0

def normalize(counts):
    """
    将一个计数器 (e.g., {'A': 1, 'B': 3}) 转换为概率字典。
    返回一个字典 (e.g., {'A': 0.25, 'B': 0.75})。
    """
    total = sum(counts.values())
    
    # 如果总数为 0 (例如一个从未见过的词)，我们无法归一化。
    if total == 0:
        return {}
        
    probabilities = {}
    for key, count in counts.items():
        probabilities[key] = count / total
    return probabilities

def normalize_with_smoothing(counts, vocabulary_size):
    """
    使用拉普拉斯平滑来归一化计数器。
    vocabulary_size (V) 是所有可能结果的数量。
    """
    total = sum(counts.values())
    
    # (count + k) / (N + k*V)
    # 这里我们使用 k=1 (加一平滑)
    denominator = total + (LAPLACE_SMOOTHING * vocabulary_size)
    
    probabilities = {}
    for key, count in counts.items():
        probabilities[key] = (count + LAPLACE_SMOOTHING) / denominator
        
    # 我们还需要一个 "__DEFAULT__" 概率，用于处理从未见过的转换
    # P(unseen) = k / (N + k*V)
    probabilities['__DEFAULT__'] = LAPLACE_SMOOTHING / denominator
    
    return probabilities

def main():
    print(f"正在从 {CORPUS_FILE} 加载语料库...")
    try:
        with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到语料库文件 '{CORPUS_FILE}'")
        print("请先运行 'python3 build_local_corpus.py' 来生成数据。")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误: {CORPUS_FILE} 文件已损坏或为空。")
        print("请尝试重新运行 'python3 build_local_corpus.py'。")
        sys.exit(1)
        
    print(f"语料库加载成功！共 {len(corpus)} 个句子。")
    print("开始训练 HMM 模型...")

    # 初始化计数器
    # 1. 发射计数 (Emission): P(汉字 | 读音)
    #    e.g., emission_counts['とうし']['投資'] = 100
    emission_counts = defaultdict(Counter)
    
    # 2. 转移计数 (Transition): P(读音_i | 读音_i-1)
    #    e.g., transition_counts['わたし']['は'] = 500
    transition_counts = defaultdict(Counter)
    
    # 3. 先验计数 (Prior): P(读音) (作为句子的第一个词)
    #    e.g., prior_counts['わたし'] = 200
    prior_counts = Counter()
    
    # 4. 读音词汇表 (用于平滑)
    reading_vocabulary = set()

    # --- 第一轮：计数 ---
    for sentence in corpus:
        if not sentence:
            continue
            
        # 1. 计数先验概率 (Prior)
        first_word_reading = sentence[0][1] # (汉字, 读音)
        prior_counts[first_word_reading] += 1
        
        prev_reading = None
        
        for (kanji, reading) in sentence:
            # 2. 计数发射概率 (Emission)
            emission_counts[kanji][reading] += 1
            
            # 3. 计数转移概率 (Transition)
            if prev_reading is not None:
                transition_counts[prev_reading][reading] += 1
                
            # 4. 更新词汇表和 prev_reading
            reading_vocabulary.add(reading)
            prev_reading = reading
            
    print("计数完成。")
    vocabulary_size = len(reading_vocabulary)
    print(f"读音词汇表大小 (V) = {vocabulary_size}")

    # --- 第二轮：规范化 (计算概率) ---
    print("正在计算概率 (规范化)...")
    
    # 1. 规范化先验概率
    priors = normalize_with_smoothing(prior_counts, vocabulary_size)
    
    # 2. 规范化转移概率
    transitions = {}
    for prev_reading, next_reading_counts in transition_counts.items():
        transitions[prev_reading] = normalize_with_smoothing(next_reading_counts, vocabulary_size)
        
    # 3. 规范化发射概率 (这里不需要平滑，因为 P(汉字|读音) 是固定的)
    emissions = {}
    for kanji, reading_counts in emission_counts.items():
        emissions[kanji] = normalize(reading_counts)
        
    print("概率计算完成。")

    # --- !!! 关键修复：将所有模型保存在一个字典中 !!! ---
    model = {
        'priors': priors,
        'transitions': transitions,
        'emissions': emissions
    }

    # --- 保存模型 ---
    try:
        with open(MODEL_FILE, 'w', encoding='utf-8') as f:
            # 确保保存的是 'model' 字典，而不是 'priors' 或其他
            json.dump(model, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存模型到 {MODEL_FILE} 时出错: {e}")
        sys.exit(1)

    print(f"\n--- 训练完成！ ---")
    print(f"模型已成功保存到 {MODEL_FILE}")
    print("你现在可以运行 'python3 yomigami.py' 来测试了！")

if __name__ == '__main__':
    main()
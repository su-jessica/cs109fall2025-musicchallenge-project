import json
import sys

# 确保 MeCab Python 绑定已安装
try:
    import MeCab
except ImportError:
    print("错误: 未安装 MeCab Python 绑定 (mecab-python3)。\n请安装 mecab-python3。")
    sys.exit(1)

# --- 配置 ---
CUSTOM_FILE = 'multi_reading_corpus_v1.json'
OUTPUT_FILE = 'corpus_custom.json'
# 确保你的 MeCab 路径是正确的
MECAB_CONFIG = "-r /opt/homebrew/etc/mecabrc -d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd"

# --- 修正后的 kata_to_hira 函数 (在 preprocess_custom.py 文件中) ---

def kata_to_hira(katakana_string):
    """将片假名转换为平假名（并移除所有空格，确保对齐）"""
    hiragana = []
    for char in katakana_string:
        code = ord(char)
        if 12449 <= code <= 12534: 
            # 这是一个片假名，转换为平假名
            hiragana.append(chr(code - 96)) 
        else:
            # 非片假名（如空格、标点）保持不变
            hiragana.append(char)
    # 关键修正：确保在返回前移除所有空格，防止在 zip() 对齐时出错
    return "".join(hiragana).replace(' ', '')

def main():
    print(f"开始预处理自定义语料库 '{CUSTOM_FILE}'...")
    
    try:
        tagger = MeCab.Tagger(MECAB_CONFIG)
    except RuntimeError:
        print("错误: MeCab 初始化失败。请检查 MECAB_CONFIG 路径是否正确。")
        sys.exit(1)
    
    final_corpus = []

    # 1. 加载文件
    try:
        with open(CUSTOM_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到自定义语料库文件 '{CUSTOM_FILE}'")
        sys.exit(1)
    
    successful_count = 0
    total_count = len(data)

    # 2. 处理数据
    for item in data:
        sentence = item['sentence']
        # 目标读音转换为平假名，并移除空格
        target_readings = [kata_to_hira(r) for r in item['reading']] 
        
        node = tagger.parseToNode(sentence)
        
        surface_words = [] 
        
        while node:
            surface = node.surface
            # 过滤掉空的 surface (如 EOS)
            if surface and surface != 'EOS':
                features = node.feature.split(',')
                # 获取 MeCab 的读音 (作为判断是否为“词语”的标准)
                reading_kata = features[7] if len(features) > 7 else '' 
                
                # 如果 MeCab 给出了读音（通常表明它是一个有意义的词），我们才记录它
                if reading_kata != '*':
                    surface_words.append(surface)
                    
            node = node.next

        # 核心检查：只检查“有读音的词语”数量
        if len(surface_words) != len(target_readings):
            # 这里的警告被注释掉，但逻辑上是跳过不匹配的句子
            continue
            
        # 成功对齐：将 MeCab 的分词结果与你 JSON 中提供的目标读音配对
        sentence_pairs = list(zip(surface_words, target_readings))
        final_corpus.append(sentence_pairs)
        successful_count += 1


    # 3. 保存文件
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
            json.dump(final_corpus, out_f, ensure_ascii=False, indent=2)
        print(f"\n--- 预处理完成！---")
        print(f"成功处理 {successful_count} / {total_count} 条句子，已保存到 '{OUTPUT_FILE}'。")
    except Exception as e:
        print(f"保存到 {OUTPUT_FILE} 时出错: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
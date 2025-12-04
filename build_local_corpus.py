import re
import json
import MeCab
import time
import sys
import os

# --- 配置 ---
DATA_FOLDER = "aozorabunko_text-master/cards"
MAX_BOOKS_TO_PROCESS = 100 

# Aozora 格式的正则表达式
RE_ANNOTATION = re.compile(r'［＃.*?］')
RE_SEPARATOR = re.compile(r'^-{5,}.*?$', re.MULTILINE)
RE_RUBY = re.compile(r'｜?([^《]+?)《(.+?)》')

def kata_to_hira(katakana_string):
    """将片假名字符串转换为平假名"""
    hiragana = []
    for char in katakana_string:
        code = ord(char)
        if 12449 <= code <= 12534: 
            hiragana.append(chr(code - 96)) 
        else:
            hiragana.append(char)
    return "".join(hiragana)

class AozoraParser:
    def __init__(self):
        try:
            # !!! 最终升级：使用 NEologd 字典 !!!
            # 我们从 -r (配置文件) 改为 -d (字典目录)
            self.tagger = MeCab.Tagger("-r /opt/homebrew/etc/mecabrc -d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd")
            
        except RuntimeError:
            print("错误: MeCab (NEologd) 未能初始化。")
            print("请确保你已成功安装 mecab-ipadic-neologd")
            print("并且上面的 -d 路径是正确的。")
            sys.exit(1)
            
    def read_book_from_local(self, filepath):
        try:
            with open(filepath, 'r', encoding='shift_jis', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"  > 读取文件失败: {filepath} ({e})")
            return None

    def clean_text(self, text):
        parts = RE_SEPARATOR.split(text)
        if len(parts) > 1:
            text = parts[1] 
        text = RE_ANNOTATION.sub('', text)
        return [line for line in text.splitlines() if line.strip()]

    def parse_line(self, line):
        answers = {}
        for match in RE_RUBY.finditer(line):
            kanji = match.group(1) 
            reading = kata_to_hira(match.group(2))
            answers[kanji] = reading
            
        clean_line = RE_RUBY.sub(r'\1', line)
        node = self.tagger.parseToNode(clean_line)
        
        sentence_pairs = []
        while node:
            surface = node.surface 
            if not surface: 
                node = node.next
                continue
                
            if surface in answers:
                sentence_pairs.append((surface, answers[surface]))
            else:
                features = node.feature.split(',')
                reading = features[7] if len(features) > 7 else '*'
                
                if reading != '*':
                    reading_hiragana = kata_to_hira(reading)
                    sentence_pairs.append((surface, reading_hiragana))
            
            node = node.next
            
        return sentence_pairs

def main():
    print(f"开始使用 Aozora (NEologd) 构建语料库 (最多 {MAX_BOOKS_TO_PROCESS} 本书)...")
    
    if not os.path.isdir(DATA_FOLDER):
        print(f"错误: 找不到数据文件夹 '{DATA_FOLDER}'")
        sys.exit(1)
        
    parser = AozoraParser()
    all_sentences = [] 
    book_count = 0

    for root, dirs, files in os.walk(DATA_FOLDER):
        if book_count >= MAX_BOOKS_TO_PROCESS:
            break
        for filename in files:
            if book_count >= MAX_BOOKS_TO_PROCESS:
                break
            if filename.endswith('.txt'):
                filepath = os.path.join(root, filename)
                print(f"--- 正在处理: 书籍 {book_count + 1}/{MAX_BOOKS_TO_PROCESS} ({filename}) ---")
                
                text = parser.read_book_from_local(filepath)
                if not text:
                    continue
                    
                clean_lines = parser.clean_text(text)
                
                for line in clean_lines:
                    sentence_pairs = parser.parse_line(line)
                    if sentence_pairs:
                        all_sentences.append(sentence_pairs)
                
                book_count += 1
                time.sleep(0.01) 

    with open('corpus_raw.json', 'w', encoding='utf-8') as f:
        json.dump(all_sentences, f, ensure_ascii=False, indent=2)

    print(f"\n--- 语料库构建完成！ ---")
    print(f"总共 {len(all_sentences)} 个句子 (来自 {book_count} 本书) 已被保存到 corpus_raw.json")
    print("你现在可以运行 'python3 train.py' 来训练模型了。")

if __name__ == '__main__':
    main()
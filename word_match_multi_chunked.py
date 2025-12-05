# word_match_multi_chunked.py
import spacy
import re
from collections import Counter
import pandas as pd
import os

# ---------- 设置 ----------
book_file = "book.txt"  # 书籍文本文件（必须）
# 每个词块包含的单词数（可根据内存/速度调整，越大越快但越吃内存）
CHUNK_WORDS = 50000

# ---------- 加载模型并禁用 heavy components ----------
# 尝试用 exclude（兼容较新版本），若不支持再用 remove_pipe
try:
    nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner"])
except TypeError:
    nlp = spacy.load("en_core_web_sm")
    for pipe in ("parser", "ner"):
        if pipe in nlp.pipe_names:
            try:
                nlp.remove_pipe(pipe)
            except Exception:
                pass

# （可选）如果你确实需要处理超长单个 doc，也可以增加 max_length，但不推荐常用：
# nlp.max_length = max(nlp.max_length, 2000000)

# ---------- 读取书籍并提取单词 ----------
if not os.path.exists(book_file):
    raise SystemExit(f"找不到 {book_file}，请确认文件存在并与脚本在同一目录。")

with open(book_file, "r", encoding="utf-8") as f:
    text = f.read()

# 只保留字母序列作为“单词”列表
words = re.findall(r"[a-zA-Z]+", text)

# ---------- 按块生成字符串（每块 CHUNK_WORDS 个单词） ----------
def iter_chunks_wordlist(wordlist, chunk_words=CHUNK_WORDS):
    for i in range(0, len(wordlist), chunk_words):
        yield " ".join(wordlist[i:i+chunk_words])

# ---------- 对每个块用 nlp.pipe 处理并收集 lemma ----------
lemmas = []
# 使用 nlp.pipe 可以更高效地批处理多个小文本
for doc in nlp.pipe(iter_chunks_wordlist(words), batch_size=4):
    for token in doc:
        if token.is_alpha:
            lemma = token.lemma_.lower()
            # 规则：副词（ADV）以 -ly 结尾 -> 尽量还原为形容词（去掉 ly）
            if token.pos_ == "ADV" and lemma.endswith("ly"):
                base = lemma[:-2]
                if len(base) > 2:
                    lemma = base
            lemmas.append(lemma)

# ---------- 统计频次 ----------
word_counts = Counter(lemmas)

# ---------- 遍历当前文件夹所有 txt 词表，除了 book.txt ----------
for file in os.listdir():
    if file.endswith(".txt") and file != book_file:
        vocab_name = os.path.splitext(file)[0]
        print(f"正在处理 {file} ...")

        with open(file, "r", encoding="utf-8") as f:
            vocab_words = set(line.strip().lower() for line in f if line.strip())

        matched_words = {word: count for word, count in word_counts.items() if word in vocab_words}

        df = pd.DataFrame(matched_words.items(), columns=["Word", "Count"]).sort_values(by="Count", ascending=False)
        if df.empty:
            df = pd.DataFrame([["无匹配结果", 0]], columns=["Word", "Count"])

        out_file = f"{vocab_name}.xlsx"
        df.to_excel(out_file, index=False)
        print(f"完成：结果已保存到 {out_file}")

print("✅ 全部处理完成！")

import streamlit as st
import spacy
import re
from collections import Counter
import pandas as pd
import io
import matplotlib.pyplot as plt  # 新增：用来画图
from wordcloud import WordCloud  # 新增：用来生成词云

# ==========================================
# 1. 页面配置 & 侧边栏设计
# ==========================================
st.set_page_config(page_title="书籍词汇大侦探", page_icon="🕵️", layout="wide") # layout="wide" 让页面变宽

# --- 侧边栏 (Sidebar) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2097/2097055.png", width=100) # 加个装饰小图标
    st.title("控制面板 ⚙️")
    st.markdown("在这里上传你的文件")
    
    st.subheader("1. 上传书籍 📖")
    book_file = st.file_uploader("选择书籍 (txt)", type=["txt"])
    
    st.subheader("2. 上传词表 📝")
    vocab_files = st.file_uploader("选择词表 (可多选)", type=["txt"], accept_multiple_files=True)
    
    st.info("提示：词云图生成可能需要几秒钟，请耐心等待。")
    
    # 放置一个开始按钮
    run_button = st.button("🚀 开始分析", type="primary", use_container_width=True)

# ==========================================
# 2. 主页面内容
# ==========================================
st.title("📚 书籍词频 & 词云可视化工具")
st.markdown("""
这个工具可以帮你分析一本英文书中，包含了多少个你指定词表里的单词。
**左侧上传文件，右侧查看炫酷的分析报告！**
""")

# ==========================================
# 3. 加载模型
# ==========================================
@st.cache_resource
def load_model():
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
    return nlp

# 预加载模型，避免点击按钮时卡顿
if 'nlp' not in st.session_state:
    with st.spinner('正在唤醒 AI 引擎...'):
        st.session_state.nlp = load_model()

# ==========================================
# 4. 核心逻辑
# ==========================================
if run_button:
    if not book_file:
        st.error("❌ 还没有上传书籍哦！请在左侧侧边栏上传。")
    elif not vocab_files:
        st.error("❌ 还没有上传词表哦！请在左侧侧边栏上传。")
    else:
        # --- A. 处理书籍 ---
        st.divider()
        status_text = st.empty() # 创建一个空位用来显示状态
        progress_bar = st.progress(0)
        
        status_text.write("⏳ 正在阅读书籍内容...")
        
        text = book_file.getvalue().decode("utf-8")
        words = re.findall(r"[a-zA-Z]+", text)
        
        CHUNK_WORDS = 50000 
        def iter_chunks_wordlist(wordlist, chunk_words=CHUNK_WORDS):
            for i in range(0, len(wordlist), chunk_words):
                yield " ".join(wordlist[i:i+chunk_words])

        lemmas = []
        total_chunks = (len(words) // CHUNK_WORDS) + 1
        current_chunk = 0
        
        nlp = st.session_state.nlp

        status_text.write("🧠 AI 正在分析单词原形...")
        for doc in nlp.pipe(iter_chunks_wordlist(words), batch_size=4):
            current_chunk += 1
            progress_bar.progress(min(current_chunk / total_chunks, 1.0))

            for token in doc:
                if token.is_alpha:
                    lemma = token.lemma_.lower()
                    if token.pos_ == "ADV" and lemma.endswith("ly"):
                        base = lemma[:-2]
                        if len(base) > 2:
                            lemma = base
                    lemmas.append(lemma)
        
        word_counts = Counter(lemmas)
        status_text.success(f"✅ 书籍处理完成！共发现 {len(word_counts)} 个唯一单词。")
        
        # --- B. 匹配与可视化 ---
        st.header("📊 分析报告")
        
        # 使用 Tabs (标签页) 来分开展示不同的词表结果
        vocab_names = [v.name for v in vocab_files]
        tabs = st.tabs(vocab_names) # 动态创建标签页

        for i, v_file in enumerate(vocab_files):
            with tabs[i]: # 在对应的标签页里画图
                vocab_name = v_file.name.split('.')[0]
                v_content = v_file.getvalue().decode("utf-8")
                vocab_words = set(line.strip().lower() for line in v_content.splitlines() if line.strip())
                
                matched_words = {word: count for word, count in word_counts.items() if word in vocab_words}
                df = pd.DataFrame(matched_words.items(), columns=["Word", "Count"]).sort_values(by="Count", ascending=False)
                
                # 布局：左边放图表，右边放数据表
                c1, c2 = st.columns([2, 1]) 
                
                with c1:
                    st.subheader(f"☁️ {vocab_name} 词云图")
                    if not df.empty:
                        # 生成词云
                        wc = WordCloud(
                            width=800, height=500, 
                            background_color='white',
                            colormap='viridis' # 颜色风格
                        ).generate_from_frequencies(matched_words)
                        
                        # 显示词云
                        fig, ax = plt.subplots()
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off") # 不显示坐标轴
                        st.pyplot(fig)
                    else:
                        st.warning("没有匹配到任何单词，无法生成词云。")

                with c2:
                    st.subheader("📋 详细数据")
                    st.dataframe(df, use_container_width=True, height=400)
                    
                    # 下载按钮
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False)
                    processed_data = output.getvalue()
                    
                    st.download_button(
                        f"📥 下载 Excel",
                        data=processed_data,
                        file_name=f"{vocab_name}_result.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
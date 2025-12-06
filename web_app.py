import streamlit as st
import spacy
import re
from collections import Counter
import pandas as pd
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests
from streamlit_lottie import st_lottie

# ==========================================
# 0. 魔法函数：加载 Lottie 动画
# ==========================================
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# ==========================================
# 1. 页面配置 & 侧边栏
# ==========================================
st.set_page_config(
    page_title="智能书籍分析引擎",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 初始化 Session State (给程序安个记忆脑) ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None # 用来存由于单词计数结果

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2097/2097055.png", width=80) 
    st.title("控制中枢 ⚙️")
    st.markdown("---")
    
    st.subheader("1. 数据源接入 📖")
    book_file = st.file_uploader("上传目标书籍 (TXT格式)", type=["txt"])
    
    st.subheader("2. 知识库对接 📝")
    vocab_files = st.file_uploader("上传参考词表 (TXT格式，可多选)", type=["txt"], accept_multiple_files=True)
    
    st.markdown("---")
    
    # 这里的按钮只负责“触发计算”
    run_button = st.button("🚀 启动分析引擎", type="primary", use_container_width=True)

# ==========================================
# 2. 主页面内容 (标题修改区)
# ==========================================

lottie_tech = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json")

col_hero_1, col_hero_2 = st.columns([1, 2])

with col_hero_1:
    if lottie_tech:
        st_lottie(lottie_tech, height=200, key="tech_anim")

with col_hero_2:
    st.markdown("""
        <h1 style='display: inline-block; margin-bottom: 0;'>智能文本数据分析平台</h1>
        <span style='font-size: 1rem; color: #808080; margin-left: 10px;'> — powered by Zeno</span>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; border-left: 5px solid #FF4B4B; margin-top: 20px;'>
        <p style='font-size: 16px; color: #FAFAFA; margin: 0;'>
        欢迎使用下一代文本洞察工具。借助先进的 NLP 技术，我们将非结构化文本转化为可视化的数据资产。
        <br><b>请在左侧控制中枢上传您的数据以开始探索。</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

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

if 'nlp' not in st.session_state:
    with st.spinner('正在初始化 AI 内核...'):
        st.session_state.nlp = load_model()

# ==========================================
# 4. 核心逻辑：触发计算
# ==========================================
# 只有点击按钮时，才进行“重计算”，并把结果存入 session_state
if run_button:
    if not book_file:
        st.error("❌ 错误：未检测到书籍数据源。请在侧边栏上传。")
    elif not vocab_files:
        st.error("❌ 错误：未检测到参考词表。请在侧边栏上传。")
    else:
        # --- A. 处理书籍 ---
        st.subheader("🟢 实时处理进度")
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        status_text.markdown("**Step 1/2: 正在解析原始文本流...**")
        
        # 读取文件
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

        status_text.markdown("**Step 2/2: AI 内核正在进行语言学特征提取 (Lemmatization)...**")
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
        
        # === 关键点：计算完存入记忆，而不是直接显示 ===
        st.session_state.analysis_results = word_counts
        
        progress_bar.empty()
        status_text.empty() # 清理掉进度文字
        st.success(f"✅ 分析完成！已存入缓存。共发现 {len(word_counts)} 个唯一词汇。")

# ==========================================
# 5. 显示逻辑：渲染结果
# ==========================================
# 只要记忆里有结果，就一直显示（不管你有没有按按钮，不管你有没有刷新）
if st.session_state.analysis_results:
    word_counts = st.session_state.analysis_results
    
    st.header("📊 数据洞察报告")
    
    # 重新读取 vocab_files (Streamlit 的 uploader 会缓存文件内容，所以是安全的)
    if vocab_files:
        vocab_names = [v.name for v in vocab_files]
        tabs = st.tabs([f"📁 {name}" for name in vocab_names])

        for i, v_file in enumerate(vocab_files):
            with tabs[i]:
                vocab_name = v_file.name.split('.')[0]
                # 每次读取前如果不重置指针，多次读取可能为空，所以用 getvalue() 最稳
                v_content = v_file.getvalue().decode("utf-8")
                vocab_words = set(line.strip().lower() for line in v_content.splitlines() if line.strip())
                
                matched_words = {word: count for word, count in word_counts.items() if word in vocab_words}
                df = pd.DataFrame(matched_words.items(), columns=["Word", "Count"]).sort_values(by="Count", ascending=False)
                
                c1, c2 = st.columns([2, 1]) 
                
                with c1:
                    st.subheader(f"☁️ {vocab_name} - 语义云图")
                    if not df.empty:
                        wc = WordCloud(
                            width=800, height=500, 
                            background_color='#0E1117',
                            colormap='plasma',
                            font_path=None
                        ).generate_from_frequencies(matched_words)
                        
                        fig, ax = plt.subplots()
                        fig.patch.set_facecolor('#0E1117')
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                    else:
                        st.warning("⚠️ 该词表中未发现任何匹配项。")

                with c2:
                    st.subheader("📋 结构化数据明细")
                    
                    if df.empty:
                        st.info("暂无数据")
                    else:
                        # 插入勾选列
                        df.insert(0, "Select", False)
                        
                        # 关键点：给 data_editor 一个唯一的 key，防止它在重绘时丢失状态
                        # 我们用 vocab_name 作为 key 的一部分
                        edited_df = st.data_editor(
                            df,
                            column_config={
                                "Select": st.column_config.CheckboxColumn(
                                    "导出?",
                                    default=False,
                                )
                            },
                            disabled=["Word", "Count"],
                            hide_index=True,
                            use_container_width=True,
                            height=400,
                            key=f"editor_{vocab_name}" 
                        )
                        
                        selected_rows = edited_df[edited_df["Select"] == True]
                        export_data = selected_rows.drop(columns=["Select"])
                        
                        st.caption(f"已选择 {len(export_data)} 个单词准备导出")
                        
                        if not export_data.empty:
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                export_data.to_excel(writer, index=False)
                            processed_data = output.getvalue()
                            
                            st.download_button(
                                f"📥 导出已选数据 (.xlsx)",
                                data=processed_data,
                                file_name=f"{vocab_name}_selected.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                                type="primary"
                            )
                        else:
                            st.download_button(
                                "📥 请先勾选单词",
                                data=b"",
                                disabled=True,
                                use_container_width=True
                            )

# ==========================================
# 6. 注入页脚
# ==========================================
footer_css = """
<style>
.footer {
    position: fixed;
    left: 20px;
    bottom: 20px;
    width: auto;
    background-color: transparent;
    color: #808080;
    text-align: left;
    z-index: 999;
    font-family: sans-serif;
    font-size: 14px;
    pointer-events: none;
}
</style>
<div class="footer">
    <p>⚡ Powered by <b>Zeno</b></p>
</div>
"""
st.markdown(footer_css, unsafe_allow_html=True)

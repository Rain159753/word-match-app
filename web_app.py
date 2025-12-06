import streamlit as st
import spacy
import re
from collections import Counter
import pandas as pd
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests  # 新增：用来下载动画文件
from streamlit_lottie import st_lottie # 新增：用来播放动画

# ==========================================
# 0. 魔法函数：加载 Lottie 动画
# ==========================================
# 这是一个通用的函数，给定一个 URL，它会把动画数据抓取下来
@st.cache_data # 加个缓存，避免每次刷新都重新下载
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ==========================================
# 1. 页面配置 & 侧边栏设计
# ==========================================
st.set_page_config(
    page_title="智能书籍分析引擎", # 改个更高大上的名字
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 侧边栏 (Sidebar) ---
with st.sidebar:
    # 这里也可以换成一个更酷的科技感 Logo 图片链接
    st.image("https://cdn-icons-png.flaticon.com/512/2097/2097055.png", width=80) 
    st.title("控制中枢 ⚙️")
    st.markdown("---") # 分割线
    
    st.subheader("1. 数据源接入 📖")
    book_file = st.file_uploader("上传目标书籍 (TXT格式)", type=["txt"])
    
    st.subheader("2. 知识库对接 📝")
    vocab_files = st.file_uploader("上传参考词表 (TXT格式，可多选)", type=["txt"], accept_multiple_files=True)
    
    st.markdown("---")
    
    # 放置一个开始按钮，加个不同颜色的提示
    run_button = st.button("🚀 启动分析引擎", type="primary", use_container_width=True)
    if run_button:
         st.caption("引擎正在预热，即将开始计算...")


# ==========================================
# 2. 主页面内容 (颜值升级区)
# ==========================================

# --- A. 头部 Hero 区域 (动画 + 标题) ---
# 加载一个酷炫的科技感 Lottie 动画 (这是一个免费的示例地址)
lottie_tech = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json")

col_hero_1, col_hero_2 = st.columns([1, 2]) # 左窄右宽

with col_hero_1:
    # 在左侧显示动画
    if lottie_tech:
        st_lottie(lottie_tech, height=200, key="tech_anim")

with col_hero_2:
    # 在右侧显示大标题
    st.title("智能文本数据分析平台")
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; border-left: 5px solid #FF4B4B;'>
        <p style='font-size: 16px; color: #FAFAFA;'>
        欢迎使用下一代文本洞察工具。借助先进的 NLP 技术，我们将非结构化文本转化为可视化的数据资产。
        <br><b>请在左侧控制中枢上传您的数据以开始探索。</b>
        </p>
    </div>
    """, unsafe_allow_html=True) # 使用了一点 HTML/CSS 来美化文字框

st.divider()

# ==========================================
# 3. 加载模型 (保持不变)
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

# 预加载模型
if 'nlp' not in st.session_state:
    # 这里用一个空的占位符，让加载过程不那么突兀
    with st.spinner('正在初始化 AI 内核...'):
        st.session_state.nlp = load_model()

# ==========================================
# 4. 核心逻辑 (逻辑不变，只微调了提示文案)
# ==========================================
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
        progress_bar.empty() # 处理完后隐藏进度条，更清爽
        # 用一个漂亮的成功提示框
        st.success(f"✅ 数据预处理完毕！成功建立索引，包含 {len(word_counts)} 个唯一词汇基元。")
        
        # --- B. 匹配与可视化 ---
        st.header("📊 数据洞察报告")
        
        vocab_names = [v.name for v in vocab_files]
        tabs = st.tabs([f"📁 {name}" for name in vocab_names]) # 给标签页加个小图标

        for i, v_file in enumerate(vocab_files):
            with tabs[i]:
                vocab_name = v_file.name.split('.')[0]
                v_content = v_file.getvalue().decode("utf-8")
                vocab_words = set(line.strip().lower() for line in v_content.splitlines() if line.strip())
                
                matched_words = {word: count for word, count in word_counts.items() if word in vocab_words}
                df = pd.DataFrame(matched_words.items(), columns=["Word", "Count"]).sort_values(by="Count", ascending=False)
                
                c1, c2 = st.columns([2, 1]) 
                
                with c1:
                    st.subheader(f"☁️ {vocab_name} - 语义云图")
                    if not df.empty:
                        # 调整了词云背景色，适应暗黑模式
                        wc = WordCloud(
                            width=800, height=500, 
                            background_color='#0E1117', # 配合暗黑背景
                            colormap='plasma', # 换个更科技感的配色
                            font_path=None # 如果有中文字体需求需指定
                        ).generate_from_frequencies(matched_words)
                        
                        fig, ax = plt.subplots()
                        fig.patch.set_facecolor('#0E1117') # 设置图片背景透明/黑色
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                    else:
                        st.warning("⚠️ 该词表中未发现任何匹配项。")

                with c2:
                    st.subheader("📋 结构化数据明细")
                    st.dataframe(df, use_container_width=True, height=400)
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False)
                    processed_data = output.getvalue()
                    
                    st.download_button(
                        f"📥 导出 {vocab_name} 数据集 (.xlsx)",
                        data=processed_data,
                        file_name=f"{vocab_name}_analysis_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        type="secondary" # 按钮样式设为次要，不抢主按钮风头
                    )

# ==========================================
# 5. 注入页脚 (魔法 CSS)
# ==========================================
# 这段是纯 HTML/CSS 代码，用来把文字固定在左下角
footer_css = """
<style>
.footer {
    position: fixed;
    left: 20px;
    bottom: 20px;
    width: auto;
    background-color: transparent;
    color: #808080; /* 灰色字体，低调一点 */
    text-align: left;
    z-index: 999; /* 保证浮在最上层 */
    font-family: sans-serif;
    font-size: 14px;
    pointer-events: none; /* 防止挡住后面的操作 */
}
</style>
<div class="footer">
    <p>⚡ Powered by <b>Gemini</b></p>
</div>
"""
# 使用 unsafe_allow_html=True 强制渲染这段代码
st.markdown(footer_css, unsafe_allow_html=True)

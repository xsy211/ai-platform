# ==========================
# 小xu同学的AI应用平台（最终优化版）
# 第30天：界面美化+功能优化+使用引导
# ==========================
import streamlit as st
from openai import OpenAI
import faiss
import numpy as np
import os

# 全局页面配置 + 基础美化
st.set_page_config(
    page_title="小xu同学的AI应用平台",
    layout="wide",
    page_icon="🤖"
)

# 自定义页面样式
st.markdown("""
<style>
.main {background-color: #f7f8fa;}
.title {color: #1f2937; text-align: center; font-size: 28px; font-weight: bold;}
.desc {color: #4b5563; text-align: center; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# 标题与简介
st.markdown('<p class="title">🤖 小xu同学的AI应用平台</p>', unsafe_allow_html=True)
st.markdown('<p class="desc">智能聊天 | 文档问答 | 文案生成 三合一工具</p>', unsafe_allow_html=True)
st.divider()

# ========= 安全读取 API Key =========
def get_api_key():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
        return os.getenv("OPENAI_API_KEY")
    except:
        return None

api_key = get_api_key()
client = None
if api_key:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )

# 模型列表
MODEL_LIST = ["deepseek-chat", "deepseek-coder"]

# 会话初始化
if "knowledge_chunks" not in st.session_state:
    st.session_state.knowledge_chunks = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# ---------------------- 侧边栏：自检 + 功能区 + 清空按钮 ----------------------
with st.sidebar:
    st.subheader("🔍 系统自检")
    if st.button("检测接口连通性"):
        if not api_key:
            st.error("❌ 未配置 API Key")
        elif not client:
            st.error("❌ 客户端初始化失败")
        else:
            try:
                test_resp = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": "只回复OK"}],
                    temperature=0,
                    max_tokens=5
                )
                if test_resp.choices[0].message.content.strip() == "OK":
                    st.success("✅ DeepSeek 接口正常")
                    st.info(f"接口地址：{client.base_url}")
            except Exception as e:
                st.error(f"❌ 连接失败：{str(e)[:80]}")

    st.divider()
    st.subheader("🧭 功能菜单")
    menu = st.radio("请选择功能", ["AI智能聊天", "文档知识库问答", "自媒体文案生成"])

    st.divider()
    st.subheader("⚙️ 模型设置")
    select_model = st.selectbox("切换大模型", MODEL_LIST)

    st.divider()
    # 新增：清空聊天记录按钮
    if st.button("🧹 清空聊天记录"):
        st.session_state.chat_messages = []
        st.success("✅ 聊天记录已清空")

    st.divider()
    # 新增：使用说明
    st.info("""
    💡 使用小贴士
    1. 聊天：直接输入对话即可
    2. 文档问答：仅支持 TXT 文件
    3. 文案生成：填写主题+风格一键产出
    """)

# ---------------------- 公共工具函数 ----------------------
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def get_embedding(text):
    resp = client.embeddings.create(
        input=text,
        model="deepseek-embedding"
    )
    return resp.data[0].embedding

def build_vector_db(file_content):
    chunks = split_text(file_content)
    embeddings = [get_embedding(chunk) for chunk in chunks]
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings, dtype=np.float32))
    return chunks, index

def search_context(query, chunks, index, top_k=3):
    query_embedding = get_embedding(query)
    _, indices = index.search(np.array([query_embedding], dtype=np.float32), top_k)
    return "\n---\n".join([chunks[i] for i in indices[0]])

# ====================== 1. AI智能聊天 ======================
if menu == "AI智能聊天":
    st.subheader("💬 在线对话助手")
    if not client:
        st.error("❌ 接口异常，请先在侧边栏完成自检！")
        st.stop()

    # 展示历史消息
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("在这里输入你的问题...")
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"), st.spinner("AI思考中..."):
            resp = client.chat.completions.create(
                model=select_model,
                messages=st.session_state.chat_messages,
                temperature=0.3
            )
            reply = resp.choices[0].message.content
            st.markdown(reply)
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})

# ====================== 2. 文档知识库问答 ======================
elif menu == "文档知识库问答":
    st.subheader("📚 私有文档问答系统")
    if not client:
        st.error("❌ 接口异常，请先在侧边栏完成自检！")
        st.stop()

    # 限制上传文件大小：5MB
    upload_file = st.file_uploader("上传 TXT 文档（最大 5MB）", type="txt")
    if upload_file and upload_file.size > 5 * 1024 * 1024:
        st.error("⚠️ 文件过大，请上传 5MB 以内的文档！")
        st.stop()

    if upload_file:
        content = upload_file.read().decode("utf-8")
        with st.spinner("解析文档并构建知识库..."):
            chunks, index = build_vector_db(content)
            st.session_state.knowledge_chunks = chunks
            st.session_state.faiss_index = index
        st.success("✅ 文档加载完成，可以开始提问！")

    if st.session_state.knowledge_chunks:
        question = st.text_input("请针对文档内容提问：")
        if question:
            with st.spinner("检索答案中..."):
                context = search_context(question, st.session_state.knowledge_chunks, st.session_state.faiss_index)
                prompt = f"""严格根据参考资料回答问题，无相关内容则如实说明。
参考资料：{context}
用户问题：{question}"""
                resp = client.chat.completions.create(
                    model=select_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                st.markdown("### 📝 回答")
                st.write(resp.choices[0].message.content)
    else:
        st.info("请先上传 TXT 格式文档")

# ====================== 3. 自媒体文案生成 ======================
elif menu == "自媒体文案生成":
    st.subheader("✍️ 一键文案生成工具")
    if not client:
        st.error("❌ 接口异常，请先在侧边栏完成自检！")
        st.stop()

    theme = st.text_input("请输入文案主题")
    style = st.selectbox(
        "选择文案风格",
        ["走心治愈", "活泼有趣", "文艺清新", "简约高级", "搞笑段子"]
    )

    if st.button("🚀 立即生成"):
        if not theme:
            st.warning("⚠️ 请先填写文案主题！")
        else:
            with st.spinner("文案创作中..."):
                prompt = f"""围绕主题「{theme}」，撰写一篇{style}风格的自媒体文案，包含标题、正文、结尾互动。"""
                resp = client.chat.completions.create(
                    model=select_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4
                )
                st.markdown("### ✨ 生成结果")
                st.write(resp.choices[0].message.content)

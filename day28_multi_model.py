# ==========================
# 多模型综合AI平台（DeepSeek适配+自检版）
# ==========================
import streamlit as st
from openai import OpenAI
import faiss
import numpy as np
import os

# 页面全局配置
st.set_page_config(page_title="多模型AI工作台", layout="wide")
st.title("🤖 多模型综合AI应用平台")

# ========= 安全读取 API Key（适配本地/云端）=========
def get_api_key():
    try:
        # 优先读取 Streamlit Secrets（云端）
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
        # 其次读取本地环境变量
        return os.getenv("OPENAI_API_KEY")
    except:
        return None

api_key = get_api_key()

# ✅ 关键修改：指定 DeepSeek 的接口地址
client = None
if api_key:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )
# ==================================================

# 可选模型（DeepSeek 官方模型名称）
MODEL_LIST = ["deepseek-chat", "deepseek-coder"]

# 会话缓存初始化
if "knowledge_chunks" not in st.session_state:
    st.session_state.knowledge_chunks = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# ---------------------- 一键自检功能 ----------------------
st.sidebar.divider()
st.sidebar.subheader("🔍 系统自检")
if st.sidebar.button("点击自检 DeepSeek 接口"):
    if not api_key:
        st.sidebar.error("❌ 未配置 API Key，请在 Secrets 中添加 OPENAI_API_KEY")
    elif not client:
        st.sidebar.error("❌ 客户端初始化失败，请检查 API Key 格式")
    else:
        try:
            # 发送一个测试请求
            test_resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "只回复我：OK"}],
                temperature=0,
                max_tokens=5
            )
            if test_resp.choices[0].message.content.strip() == "OK":
                st.sidebar.success("✅ DeepSeek 接口连接正常！")
                st.sidebar.info(f"当前接口地址：{client.base_url}")
                st.sidebar.info(f"API Key 前5位：{api_key[:5]}...")
            else:
                st.sidebar.warning("⚠️ 接口有响应，但返回异常，请检查模型配置")
        except Exception as e:
            st.sidebar.error(f"❌ 连接失败：{str(e)[:100]}...")

# 侧边栏
with st.sidebar:
    st.divider()
    st.subheader("功能菜单")
    menu = st.radio("选择功能", ["AI智能聊天", "文档知识库问答", "自媒体文案生成"])
    st.divider()
    st.subheader("模型选择")
    select_model = st.selectbox("切换模型", MODEL_LIST)

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
    # 使用 DeepSeek 的嵌入模型
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
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), top_k)
    return "\n---\n".join([chunks[i] for i in indices[0]])

# ====================== 1. AI智能聊天 ======================
if menu == "AI智能聊天":
    st.subheader("💬 在线对话助手")
    # 显示历史消息
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入框
    user_input = st.chat_input("输入你的问题...")
    if user_input:
        if not client:
            st.error("❌ API Key 未配置或接口连接失败，请先在侧边栏完成自检")
            st.stop()
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"), st.spinner("思考中..."):
            resp = client.chat.completions.create(
                model=select_model,
                messages=[{"role": "user", "content": user_input}],
                temperature=0.3
            )
            reply = resp.choices[0].message.content
            st.markdown(reply)
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})

# ====================== 2. 文档知识库问答 ======================
elif menu == "文档知识库问答":
    st.subheader("📚 私有文档问答系统")
    if not client:
        st.error("❌ API Key 未配置或接口连接失败，请先在侧边栏完成自检")
        st.stop()
    upload_file = st.file_uploader("上传TXT文档", type="txt")

    if upload_file:
        content = upload_file.read().decode("utf-8")
        with st.spinner("构建知识库..."):
            chunks, index = build_vector_db(content)
            st.session_state.knowledge_chunks = chunks
            st.session_state.faiss_index = index
        st.success("✅ 文档加载完成，可开始提问")

    if st.session_state.knowledge_chunks:
        question = st.text_input("请输入你的问题：")
        if question:
            with st.spinner("检索答案中..."):
                context = search_context(question, st.session_state.knowledge_chunks, st.session_state.faiss_index)
                prompt = f"""根据以下参考信息回答问题，如信息不足请说明。
参考信息：{context}
问题：{question}"""
                resp = client.chat.completions.create(
                    model=select_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                answer = resp.choices[0].message.content
                st.markdown("### 回答")
                st.write(answer)
    else:
        st.info("请先上传TXT文档")

# ====================== 3. 自媒体文案生成 ======================
elif menu == "自媒体文案生成":
    st.subheader("✍️ 一键生成文案工具")
    if not client:
        st.error("❌ API Key 未配置或接口连接失败，请先在侧边栏完成自检")
        st.stop()
    theme = st.text_input("文案主题（如：雪山、咖啡、旅行）")
    style = st.selectbox("选择风格", ["走心治愈", "活泼有趣", "文艺清新", "简约高级", "搞笑段子"])

    if st.button("一键生成"):
        if not theme:
            st.warning("请输入文案主题")
        else:
            with st.spinner("生成中..."):
                prompt = f"""请根据主题「{theme}」，写一篇{style}风格的自媒体文案，结构包含标题、正文、结尾，适合朋友圈/小红书发布。"""
                resp = client.chat.completions.create(
                    model=select_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4
                )
                result = resp.choices[0].message.content
                st.markdown("### 生成结果")
                st.write(result)

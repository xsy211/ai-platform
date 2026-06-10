# ==========================
# 多模型综合AI平台（永久部署安全版）
# ==========================
import streamlit as st
from openai import OpenAI
import faiss
import numpy as np
import os

# 页面全局配置
st.set_page_config(page_title="多模型AI工作台", layout="wide")
st.title("🤖 多模型综合AI应用平台")

# ========= 安全读取 API Key（本地/云端自动适配）=========
def get_api_key():
    try:
        # 云端读取
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
        # 本地读取
        return os.getenv("OPENAI_API_KEY")
    except:
        st.error("❌ 未配置 OPENAI_API_KEY")
        st.stop()

api_key = get_api_key()
client = OpenAI(api_key=api_key)
# ======================================================

# 可选模型
MODEL_LIST = ["deepseek-chat", "deepseek-coder"]

# 会话缓存初始化
if "knowledge_chunks" not in st.session_state:
    st.session_state.knowledge_chunks = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chat_msg" not in st.session_state:
    st.session_state.chat_msg = []

# 侧边栏
with st.sidebar:
    st.subheader("功能菜单")
    menu = st.radio("选择功能", ["AI智能聊天", "文档知识库问答", "自媒体文案生成"])
    st.divider()
    st.subheader("模型选择")
    select_model = st.selectbox("切换大模型", MODEL_LIST)

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
        model="text-embedding-v1"
    )
    return resp.data[0].embedding

def build_vector_db(file_content):
    chunks = split_text(file_content)
    emb_list = []
    for c in chunks:
        emb_list.append(get_embedding(c))
    index = faiss.IndexFlatL2(len(emb_list[0]))
    index.add(np.array(emb_list, dtype=np.float32))
    return chunks, index

def search_context(query, chunks, index, top_k=3):
    q_emb = get_embedding(query)
    _, idx = index.search(np.array([q_emb], dtype=np.float32), top_k)
    return "\n---\n".join([chunks[i] for i in idx[0]])

# ====================== 1. AI智能聊天 ======================
if menu == "AI智能聊天":
    st.subheader("💬 在线对话助手")
    for msg in st.session_state.chat_msg:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("输入问题开始对话...")
    if user_input:
        st.session_state.chat_msg.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"), st.spinner("思考中..."):
            res = client.chat.completions.create(
                model=select_model,
                messages=st.session_state.chat_msg,
                temperature=0.3
            )
            ans = res.choices[0].message.content
            st.markdown(ans)
        st.session_state.chat_msg.append({"role": "assistant", "content": ans})

# ====================== 2. 文档知识库问答 ======================
elif menu == "文档知识库问答":
    st.subheader("📚 私有文档问答系统")
    upload_file = st.file_uploader("上传TXT文档", type="txt")

    if upload_file:
        content = upload_file.read().decode("utf-8")
        with st.spinner("解析文档并构建知识库..."):
            chunks, index = build_vector_db(content)
            st.session_state.knowledge_chunks = chunks
            st.session_state.faiss_index = index
        st.success("✅ 文档加载完成！")

    if st.session_state.knowledge_chunks:
        question = st.text_input("请提问文档相关问题：")
        if question:
            with st.spinner("检索答案中..."):
                context = search_context(question, st.session_state.knowledge_chunks, st.session_state.faiss_index)
                prompt = f"""严格根据参考资料回答问题，无相关内容则回复：未查询到相关信息。
参考资料：{context}
用户问题：{question}"""
                res = client.chat.completions.create(
                    model=select_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                st.markdown("### 回答")
                st.write(res.choices[0].message.content)
    else:
        st.info("请先上传TXT格式文档")

# ====================== 3. 自媒体文案生成 ======================
elif menu == "自媒体文案生成":
    st.subheader("✍️ AI文案生成工具")
    topic = st.text_input("文案主题")
    style = st.selectbox("选择风格", ["干货科普", "幽默口语", "走心治愈", "营销带货", "短视频口播"])

    if st.button("一键生成"):
        if not topic:
            st.warning("请填写文案主题！")
        else:
            with st.spinner("生成文案中..."):
                prompt = f"""撰写一篇自媒体文案，主题：{topic}，风格：{style}
结构要求：吸引人的标题 + 开篇引入 + 分段正文 + 结尾互动"""
                res = client.chat.completions.create(
                    model=select_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4
                )
                st.markdown("### 生成结果")
                st.write(res.choices[0].message.content)
                st.write(res.choices[0].message.content)
import os
import streamlit as st
import chromadb
import numpy as np
import re
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. 环境与配置初始化 ---
load_dotenv()

st.set_page_config(page_title="复初企业AI转型咨询助手", layout="wide")
st.title("🤖 复初企业AI转型知识咨询系统 (公域生产版)")

# 初始化独一无二的 Session ID，确保不同浏览器标签页（不同用户）的数据完全独立
if "user_session_id" not in st.session_state:
    import uuid
    st.session_state.user_session_id = str(uuid.uuid4())

# --- 2. 统一的云端 API 客户端初始化 ---

@st.cache_resource
def get_llm_client(api_key: str):
    """DeepSeek 大模型客户端"""
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

@st.cache_resource
def get_embedding_client():
    """云端向量化/重排客户端（以硅基流动为例）"""
    api_key = os.getenv("sk-xkdxajipiwptbupuvnbmswcvuunrvlsbzwltjktnfnjgkjiz", "请在此处或环境变量中填写Key")
    return OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

@st.cache_resource
def init_vector_db():
    """磁盘持久化向量数据库"""
    client = chromadb.PersistentClient(path="./chroma_db")
    return client

db_client = init_vector_db()
embedding_client = get_embedding_client()

# 动态创建/获取当前用户专属的 Collection
collection_name = f"user_{st.session_state.user_session_id.replace('-', '_')}"
chromadb_collection = db_client.get_or_create_collection(name=collection_name)


# --- 3. 核心 RAG 功能函数 ---

def split_into_chunks(content: str) -> List[str]:
    """文本切分逻辑"""
    cleaned_content = re.sub(r'\s*', '', content)
    chunks = re.split(r'(?=【\d{4}\.\d{1,2}\.\d{1,2}.*?】)', cleaned_content)
    final_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return final_chunks

def embed_chunks_via_api(chunks: List[str]) -> list:
    """调用在线 API 生成文本向量"""
    try:
        response = embedding_client.embeddings.create(
            model="BAAI/bge-m3",
            input=chunks
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        st.error(f"❌ 向量化失败，请检查 Embedding API 配置: {e}")
        return []

def retrieve(query: str, top_k: int) -> List[dict]:
    """检索函数"""
    query_embeddings = embed_chunks_via_api([query])
    if not query_embeddings:
        return []
        
    results = chromadb_collection.query(
        query_embeddings=[query_embeddings[0]],
        n_results=top_k
    )
    
    documents = results.get('documents') or []
    metadatas = results.get('metadatas') or []
    documents = documents[0] if len(documents) > 0 and documents[0] is not None else []
    metadatas = metadatas[0] if len(metadatas) > 0 and metadatas[0] is not None else []
    
    retrieved_data = []
    for doc, meta in zip(documents, metadatas):
        retrieved_data.append({
            "text": doc,
            "url": meta.get("url", "无") if meta else "无"
        })
    return retrieved_data

def rerank_via_api(query: str, retrieved_items: List[dict], top_k: int) -> List[dict]:
    """调用在线 Rerank API"""
    if not retrieved_items: 
        return []
    
    documents_to_rank = [item["text"] for item in retrieved_items]
    
    try:
        response = embedding_client.post(
            "/rerank",
            json={
                "model": "BAAI/bge-reranker-v2-m3",
                "query": query,
                "documents": documents_to_rank,
                "top_n": top_k
            }
        ).json()
        
        results = response.get("results", [])
        reranked_items = []
        for res in results:
            idx = res["index"]
            reranked_items.append(retrieved_items[idx])
        return reranked_items
    except Exception as e:
        st.warning(f"⚠️ Rerank 失败，已降级使用初筛顺序。错误: {e}")
        return retrieved_items[:top_k]

def generate_answer(query: str, chunks: list[str], client: OpenAI) -> str:
    """调用 DeepSeek 思考模型生成最终答案"""
    # 1. 先在外层把参考文档合并好，不用任何大括号或复杂拼接
    context_text = "\n\n".join(chunks)
    
    # 2. 严格使用三双引号开头，三双引号结尾（彻底杜绝单双引号不匹配导致的闭合错误）
    # 3. 放弃在大括号里写任何逻辑，改用最传统的 .format() 动态注入变量
    system_prompt = """你是一个专业的企业AI转型资讯顾问，需要根据用户的问题和提供的参考文档回答用户的问题，给出合理的资讯建议。
用户问题：{}
参考文档：{}
要求：
1. 尽量使用参考文档里的信息回答。
2. 回答要清晰、有条理，不要编造信息。
3. 标注引用的参考文档。""".format(query, context_text)

    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content or "抱歉，我无法生成答案。"
    except Exception as e:
        return "❌ 调用 DeepSeek 失败: " + str(e)


# --- 4. Streamlit 前端交互界面 ---
with st.sidebar:
    st.header("⚙️ 配置中心")
    api_key = st.text_input("DeepSeek API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    st.caption(f"🔒 您的专属会话隔离ID: {st.session_state.user_session_id[:8]}...")
    
    st.divider()
    
    st.header("📁 文档入库")
    uploaded_file = st.file_uploader("上传企业资讯文档 (TXT/MD)", type=['txt', 'md'])
    
    if uploaded_file and st.button("开始解析并入库"):
        file_bytes = uploaded_file.getvalue()
        try:
            content = file_bytes.decode('utf-16')
        except UnicodeDecodeError:
            try:
                content = file_bytes.decode('gbk')
            except Exception as e:
                st.error(f"❌ 文件解码失败：{e}")
                st.stop()

        if not content.strip():
           st.warning("⚠️ 读取到的文件内容为空。")
           st.stop()

        chunks = split_into_chunks(content)
       
        with st.spinner("正在通过云端 API 生成向量并安全存入隔离库..."):
            embeddings = embed_chunks_via_api(chunks)
            if not embeddings:
                st.stop()
                
            ids = [f"{st.session_state.user_session_id}_{i}" for i in range(len(chunks))]
            
            metadatas = []
            for chunk in chunks:
                urls = re.findall(r'(https://mp.weixin.qq.com/s/[^\s\)\"\'\>]+)', chunk)
                url_str = urls[0] if urls else "无"
                metadatas.append({"url": url_str})

            chromadb_collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas        
            )
        st.success(f"入库成功！您的专属知识库已存入 {len(chunks)} 个完整知识板块。")

# 主界面：对话历史渲染
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("询问关于企业AI转型的问题..."):
    if not api_key:
        st.error("请先在左侧配置中心输入你的 DeepSeek API Key")
    else:
        llm_client = get_llm_client(api_key)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("正在思考...", expanded=True) as status:
                st.write("正在跨文档智能检索...")
                initial_items = retrieve(prompt, top_k=15) 
                
                st.write("正在精细化二次重排 (Reranking)...")
                final_items = rerank_via_api(prompt, initial_items, top_k=3) 
                
                st.write("正在组织语言并调用 DeepSeek-R1 深度思考...")
                final_chunks_text = [item["text"] for item in final_items]
                answer = generate_answer(prompt, final_chunks_text, llm_client)
                
                status.update(label="方案生成完毕！", state="complete", expanded=False)

            st.markdown(answer)
            
            with st.expander("查看本次回答参考的原始知识切片"):
                for i, item in enumerate(final_items):
                    st.info(f"参考内容 {i+1}:\n\n{item['text']}")
                    if item['url'] != "无":
                        st.markdown(f"🔗 **文章原链:** [{item['url']}]({item['url']})")

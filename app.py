import os
import streamlit as st
import chromadb
import numpy as np
import re
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# --- 1. 环境与配置初始化 ---
load_dotenv()

st.set_page_config(page_title="AI领导力洞察专属咨询系统", layout="wide")
st.title("🤖 AI领导力洞察专属咨询系统 (唯一文档固化版)")

# --- 2. 统一的云端 API 客户端与固定知识库初始化 ---

@st.cache_resource
def get_llm_client(api_key: str):
    """DeepSeek 大模型客户端"""
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

@st.cache_resource
def get_local_embedding_model():
    """在 Streamlit 服务器本地直接加载向量模型（零成本、零API限制）"""
    # 首次启动会自动下载该超轻量顶级中文向量模型（约几百MB），后续直接秒级驻留内存
    return SentenceTransformer('BAAI/bge-small-zh-v1.5')

local_embed_model = get_local_embedding_model()

# 全局初始化客户端
embedding_client = get_embedding_client()

def split_into_chunks(content: str) -> List[str]:
    """文本切分逻辑"""
    cleaned_content = re.sub(r'\s*', '', content)
    chunks = re.split(r'(?=【\d{4}\.\d{1,2}\.\d{1,2}.*?】)', cleaned_content)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

@st.cache_resource
def init_fixed_vector_db():
    """磁盘持久化向量数据库（带云端自动兜底初始化功能）"""
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # 使用 get_or_create_collection 确保即使不存在也不会触发 ValueError 崩溃
    collection = client.get_or_create_collection(name="fixed_knowledge")
    
    # 【核心防御】如果发现集合是空的，说明 GitHub 上没传成功数据，直接在云端现场做一次自动无感入库
    if collection.count() == 0:
        filename = "AI领导力洞察-社区简版.md"
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-16") as f:
                content = f.read()
            chunks = split_into_chunks(content)
            
            # 云端实时生成向量
            response = embedding_client.embeddings.create(
                model="BAAI/bge-m3",
                input=chunks
            )
            embeddings = [item.embedding for item in response.data]
            
            ids = [f"doc_{i}" for i in range(len(chunks))]
            metadatas = []
            for chunk in chunks:
                urls = re.findall(r'(https://mp.weixin.qq.com/s/[^\s\)\"\'\>]+)', chunk)
                url_str = urls[0] if urls else "无"
                metadatas.append({"url": url_str})
                
            collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
            
    return collection

# 激活加载数据库
chromadb_collection = init_fixed_vector_db()


# --- 3. 核心 RAG 功能函数 ---

def embed_chunks_locally(chunks: List[str]) -> list:
    """本地直接计算向量，再也不用调用外部 API"""
    if not chunks:
        return []
    # 直接调用本地 CPU/GPU 进行向量化计算
    embeddings = local_embed_model.encode(chunks, normalize_embeddings=True)
    # 转换为 ChromaDB 要求的 list 格式
    return [e.tolist() for e in embeddings]

def retrieve(query: str, top_k: int) -> List[dict]:
    """从固化的专属库中检索知识"""
    if chromadb_collection.count() == 0:
        st.error("⚠️ 固定向量库数据为空，且未在根目录下找到【AI领导力洞察-社区简版.md】文本文件！")
        return []

    query_embedding = embed_query_via_api(query)
    if not query_embedding:
        return []
        
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
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
    context_text = "\n\n".join(chunks) if chunks else "未检索到相关参考文档。"
    
    system_prompt = """你是一个专业的AI领导力咨询顾问，需要根据用户的问题和提供的参考文档回答用户的问题。
用户问题：{}
参考文档：{}
要求：
1. 严格并尽量仅使用参考文档里的信息回答。
2. 回答要清晰、有条理，不要编造任何非参考文档里的信息。
3. 标注引用的参考文档里的时间戳。""".format(query, context_text)

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
    st.divider()
    st.success("✅ 《AI领导力洞察-社区简版》专属知识库已在线激活，随时可以提问。")

# 主界面：对话历史渲染
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("输入关于《AI领导力洞察》的问题..."):
    if not api_key:
        st.error("请先在左侧配置中心输入你的 DeepSeek API Key")
    else:
        llm_client = get_llm_client(api_key)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("正在智能检索与深度思考...", expanded=True) as status:
                st.write("正在从固化文档中精细检索相关切片...")
                initial_items = retrieve(prompt, top_k=15) 
                
                st.write("正在进行二次重排 (Reranking)...")
                final_items = rerank_via_api(prompt, initial_items, top_k=3) 
                
                st.write("正在调用 DeepSeek-R1 深度思考中...")
                final_chunks_text = [item["text"] for item in final_items]
                answer = generate_answer(prompt, final_chunks_text, llm_client)
                
                status.update(label="解答生成完毕！", state="complete", expanded=False)

            st.markdown(answer)
            
            with st.expander("查看本次回答参考的固定知识切片"):
                for i, item in enumerate(final_items):
                    st.info(f"参考内容 {i+1}:\n\n{item['text']}")
                    if item['url'] != "无":
                        st.markdown(f"🔗 **文章原链:** [{item['url']}]({item['url']})")

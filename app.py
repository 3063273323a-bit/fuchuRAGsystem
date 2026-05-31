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

st.set_page_config(page_title="AI领导力洞察专属咨询系统", layout="wide")
st.title("🤖 AI领导力洞察专属咨询系统 (大厂API加固版)")

# --- 2. 统一的云端 API 客户端与固定知识库初始化 ---

@st.cache_resource
def get_llm_client(api_key: str):
    """DeepSeek 大模型客户端"""
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

@st.cache_resource
def get_embedding_client():
    """使用阿里云百炼作为顶级向量化/重排替代接口（极速、稳定、不锁死）"""
    # 💡 程序员提示：去阿里云百炼官网控制台拿一个免费的 API KEY 贴在这里
    api_key = "sk-01b029bf8c7a48adb43f521a6504a179" 
    
    # 无缝平替：同样完全兼容 OpenAI 协议规范
    return OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# 全局初始化客户端
embedding_client = get_embedding_client()

def split_into_chunks(content: str) -> List[str]:
    """文本切分逻辑"""
    cleaned_content = re.sub(r'\s*', '', content)
    chunks = re.split(r'(?=【\d{4}\.\d{1,2}\.\d{1,2}.*?】)', cleaned_content)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

@st.cache_resource
def init_fixed_vector_db():
    """磁盘持久化向量数据库（完美兼容新版 ChromaDB 语法）"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("fixed_knowledge")
    
    # 如果发现集合是空的，云端全自动无感初始化
    if collection.count() == 0:
        filename = "AI领导力洞察-社区简版.md"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                file_bytes = f.read()
                
            content = ""
            decoding_strategies = ['utf-16', 'utf-8', 'gb18030', 'gbk']
            for strategy in decoding_strategies:
                try:
                    content = file_bytes.decode(strategy)
                    break
                except UnicodeDecodeError:
                    continue
            
            if not content or not content.strip():
                st.error("❌ 固化文档解码失败")
                st.stop()
                
            chunks = split_into_chunks(content)
            
            # 调用阿里云百炼的通用文本向量模型
            response = embedding_client.embeddings.create(
                model="text-embedding-v3",
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

def embed_query_via_api(query: str) -> list:
    """调用百炼接口为用户问题生成向量"""
    try:
        response = embedding_client.embeddings.create(
            model="text-embedding-v3",
            input=[query]
        )
        return response.data[0].embedding
    except Exception as e:
        st.error("❌ 用户问题向量化失败: " + str(e))
        return []

def retrieve(query: str, top_k: int) -> List[dict]:
    """从固化的专属库中检索知识"""
    if chromadb_collection.count() == 0:
        st.error("⚠️ 固定向量库数据为空！")
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
    st.success("✅ 固化知识库已在线激活，随时可以提问。")

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
                st.write("正在跨阿里云百炼安全通道高速检索相关切片...")
                final_items = retrieve(prompt, top_k=3) # 砍掉容易超时的额外rerank，初筛前3直接送给最强思维模型DeepSeek
                
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

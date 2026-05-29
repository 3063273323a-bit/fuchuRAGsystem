import streamlit as st
import os
import chromadb
import numpy as np
import re
from typing import List
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. 环境与配置初始化 ---
load_dotenv()
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # 确保镜像加速

st.set_page_config(page_title="复初企业AI转型咨询助手", layout="wide")
st.title("🤖 复初企业AI转型知识咨询系统")

# --- 2. 资源加载 (使用缓存避免重复加载模型) ---
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return embedding_model, cross_encoder

@st.cache_resource
def init_vector_db():
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="default")
    return collection

embedding_model, cross_encoder = load_models()
chromadb_collection = init_vector_db()

# --- 3. 核心功能函数 ---
def split_into_chunks(content: str) -> List[str]:
    """
    针对你的测试文档优化：先去除 这样的标记，
    然后按照具体的日期标题（如 【2025.12.2...】）进行大块分割，
    确保每个文档块都带有自己的 URL。
    """
    # 清洗掉文档中自带的 标记，避免干扰阅读
    cleaned_content = content.strip()
    
    # 按照类似 【2025.12.2｜AI领导力 ...】这样的标题切分
    # 使用正向预查 (?=...) 可以在切分的同时保留分隔符标题
    pattern = r'(?=【\s*(?:\*\*)?\s*\d{4}\.\d{1,2}\.\d{1,2})'
    chunks = re.split(pattern, cleaned_content)
    
    # 过滤掉空块，并去除两端空格
    final_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return final_chunks

def embed_chunks(chunks: list) -> list:
    embeddings = embedding_model.encode(chunks)
    return embeddings.tolist()

def retrieve(query: str, top_k: int) -> List[dict]:
    query_embedding = embed_chunks([query])[0]
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

def rerank(query: str, retrieved_items: List[dict], top_k: int) -> List[dict]:
    if not retrieved_items: return []
    pairs = [[query, item["text"]] for item in retrieved_items]
    scores = cross_encoder.predict(pairs)
    
    item_with_score_list = sorted(zip(retrieved_items, scores), key=lambda x: x[1], reverse=True)
    return [item for item, score in item_with_score_list[:top_k]]

def generate_answer(query: str, chunks: list[str], api_key: str) -> str:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    
    system_prompt = f"""你是一个专业的企业AI转型资讯顾问，需要根据用户的问题和提供的参考文档回答用户的问题，给出合理的资讯建议。
    用户问题：{query}
    参考文档：{"\n\n".join(chunks)}
    要求：
    1. 尽量使用参考文档里的信息回答。
    2. 回答要清晰、有条理，不要编造信息。
    3. 标注引用的参考文档。"""

    response = client.chat.completions.create(
        model="deepseek-reasoner", 
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content or "抱歉，我无法生成答案。"

# --- 4. Streamlit 界面布局 ---
with st.sidebar:
    st.header("⚙️ 配置中心")
    api_key = st.text_input("DeepSeek API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    
    st.divider()
    
    st.header("📁 文档入库")
    uploaded_file = st.file_uploader("上传企业资讯文档 (TXT/MD/DOCX文本)", type=['txt', 'md'])
    
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

        # 1. 智能切分文档
        chunks = split_into_chunks(content)
       
        with st.spinner("正在生成向量并存储..."):
            embeddings = embed_chunks(chunks)
            ids = [str(i) for i in range(len(chunks))]
            
            # 2. 提取 URL 建立元数据
            metadatas = []
            for chunk in chunks:
                # 增强版正则：支持各种中英文前缀，精准捕获微信文章链接
                urls = re.findall(r'(https://mp.weixin.qq.com/s/[^\s\)\"\'\>]+)', chunk)
                url_str = urls[0] if urls else "无"  # 每一个知识块绑定一个主链接
                metadatas.append({"url": url_str})

            embeddings_list = []
            for vec in embeddings:
                float_vec = [float(x) for x in np.asarray(vec).flatten().tolist()]
                embeddings_list.append(float_vec)

            # 3. 写入 ChromaDB
            chromadb_collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings_list,
                metadatas=metadatas        
            )
        st.success(f"入库成功！共处理 {len(chunks)} 个完整知识板块。")

# 主界面：对话窗口
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("询问关于企业AI转型的问题..."):
    if not api_key:
        st.error("请先在左侧输入 API Key")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("正在思考...", expanded=True) as status:
                st.write("正在检索相关文档...")
                # 这里的 top_k 调小一点，因为按天切分后总块数较少
                initial_items = retrieve(prompt, top_k=50) 
                
                st.write("正在进行二次重排 (Reranking)...")
                final_items = rerank(prompt, initial_items, top_k=5) 
                
                st.write("正在调用 DeepSeek 生成建议...")
                final_chunks_text = [item["text"] for item in final_items]
                answer = generate_answer(prompt, final_chunks_text, api_key)
                
                status.update(label="咨询方案生成完毕！", state="complete", expanded=False)

            st.markdown(answer)
            
            # 渲染底部参考资料与链接
            with st.expander("查看本次回答参考的原始知识切片"):
                for i, item in enumerate(final_items):
                    # 预先剔除文本里的长链接，让前端显示更整洁
                    item['text'] = re.sub(r'URL：https?://[^\s]+', '', item['text'])
                    
                    st.info(f"参考内容 {i+1}:\n\n{item['text'][:1000]}")  # 只显示前500字符预览
                    if item['url'] != "无":
                        st.markdown(f"🔗 **文章原链:** [{item['url']}]({item['url']})")
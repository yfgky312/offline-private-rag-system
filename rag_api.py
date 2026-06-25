import os
import logging
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from rank_bm25 import BM25Okapi



# 加载环境变量
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 配置日志
logging.basicConfig(level=logging.INFO)

# 初始化
app = FastAPI(title="RAG 问答 API", description="完整RAG流程：分块+向量检索+生成")



# 请求/响应模型
class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    status: str
    sources: list = []  # 新增：引用原文


# ==================== 文档解析 ====================

def parse_file_content(file) -> str:
    """根据文件扩展名解析内容"""
    filename = file.filename.lower()
    content = ""

    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        if filename.endswith('.txt'):
            with open(tmp_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif filename.endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            content = "\n".join([d.page_content for d in docs])
        else:
            raise HTTPException(status_code=400, detail=f"不支持的文件格式: {filename}")
    finally:
        os.unlink(tmp_path)  # 删除临时文件

    return content


# ==================== RAG 核心流程 ====================

def build_rag_pipeline(content: str, question: str):
    """完整RAG流程：分块 → 向量化 → 检索 → 生成 → 返回溯源"""

    logging.info(f"原始文档长度: {len(content)} 字符")

    # 1. 分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " "]
    )
    chunks = text_splitter.split_text(content)
    logging.info(f"分块数量: {len(chunks)}")

    if not chunks:
        return {
            "answer": "文档内容为空，无法回答。",
            "sources": []
        }

    # 如果文档很短，直接调用模型
    if len(content) < 1000:
        logging.info("文档较短，直接调用模型")
        answer = call_llm(question, content[:3000])
        return {
            "answer": answer,
            "sources": [content[:500]]
        }

    # ===== 关键词检索 (BM25) =====
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_question = question.split()
    bm25_scores = bm25.get_scores(tokenized_question)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
    bm25_docs = [chunks[i] for i in top_bm25_indices]

    # ===== 向量检索 =====
    retrieved_docs = []
    try:
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectorstore = Chroma.from_texts(chunks, embeddings)

        retrieved_docs = vectorstore.similarity_search(question, k=10)
        logging.info(f"向量检索到 {len(retrieved_docs)} 个相关片段")

    except Exception as e:
        logging.error(f"向量检索失败: {e}")

    # ===== 合并去重 =====
    combined_docs = []
    seen = set()
    for doc in retrieved_docs + bm25_docs:
        doc_content = doc.page_content if hasattr(doc, 'page_content') else doc
        if doc_content not in seen:
            seen.add(doc_content)
            combined_docs.append(doc)

    # ===== Rerank 精排 =====
    if combined_docs:
        try:
            from sentence_transformers import CrossEncoder
            import numpy as np

            reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512)
            pairs = [[question, doc.page_content if hasattr(doc, 'page_content') else doc] for doc in combined_docs]
            scores = reranker.predict(pairs)
            ranked_indices = np.argsort(scores)[::-1]
            combined_docs = [combined_docs[i] for i in ranked_indices]
            logging.info(f"Rerank 完成，共 {len(combined_docs)} 个候选")
        except Exception as e:
            logging.warning(f"Rerank 失败，使用原始顺序: {e}")

    # 如果合并后没有结果，改用全文
    if not combined_docs:
        logging.warning("检索结果为空，改用全文")
        context = content[:3000]
        sources = [context[:500]]
    else:
        context = "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else doc for doc in combined_docs[:5]])
        sources = [doc.page_content if hasattr(doc, 'page_content') else doc for doc in combined_docs[:5]]

    # 生成回答
    answer = call_llm(question, context)

    return {
        "answer": answer,
        "sources": sources
    }


from openai import OpenAI


def call_llm(question: str, context: str) -> str:
    # 连接本地 Ollama 服务
    client = OpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1"
    )

    if len(context) > 4000:
        context = context[:4000] + "\n...(内容已截断)"

    response = client.chat.completions.create(
        model="deepseek-r1:8b",  # 你下载的本地模型
        messages=[
            {"role": "system", "content": "基于提供的文档内容回答问题。不要编造。"},
            {"role": "user", "content": f"文档内容：\n{context}\n\n问题：{question}"}
        ]
    )
    return response.choices[0].message.content


# ==================== API 接口 ====================

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask_with_file", response_model=AnswerResponse)
async def ask_with_file(
        file: UploadFile = File(..., description="上传文档（支持 .txt, .pdf）"),
        question: str = ""
):
    try:
        logging.info(f"收到文件: {file.filename}, 问题: {question[:30]}...")

        content = parse_file_content(file)
        if not content or len(content.strip()) < 10:
            raise HTTPException(status_code=400, detail="文档内容为空或太少")

        if not question:
            question = "这份文档主要讲了什么？"

        result = build_rag_pipeline(content, question)

        return AnswerResponse(
            answer=result["answer"],
            status="success",
            sources=result["sources"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
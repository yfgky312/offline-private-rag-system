# 企业级私有化 RAG 知识库问答系统

基于本地大模型的完全离线 RAG 系统，支持混合检索、问答溯源、Docker 一键部署。

---

## 📌 项目定位

本项目是一个**企业级私有化知识库问答系统**，核心特点是：

- 🔒 **完全离线运行**：不依赖任何云端 API，数据不出内网
- 🧠 **本地大模型**：基于 Ollama + DeepSeek 8B，无需 GPU 也可运行
- 🔍 **混合检索**：向量检索 + BM25 关键词检索，兼顾语义和精确匹配
- 📎 **问答溯源**：每个回答都附带原文出处，可追溯可验证
- 🐳 **Docker 部署**：一键启动，开箱即用

---

## 🛠️ 技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 大模型 | DeepSeek 8B (Ollama) | 本地部署，完全离线 |
| Embedding | BAAI/bge-small-zh-v1.5 | 本地运行，中文优化 |
| 向量数据库 | Chroma | 轻量级，快速验证 |
| 混合检索 | 向量检索 + BM25 | 语义+关键词双路召回 |
| Web 框架 | FastAPI + Uvicorn | API 服务 |
| 前端界面 | Streamlit | 可视化交互 |
| 容器化 | Docker | 一键部署 |
| 语言 | Python 3.12 | |

---

## 📁 项目结构

offline-private-rag-system/
├── rag_api.py              # 核心 API 服务
├── app.py                  # Streamlit 前端界面
├── requirements.txt        # Python 依赖
├── Dockerfile              # Docker 镜像构建文件
├── bge-small-zh-v1.5/      # 本地 Embedding 模型
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer.json
└── README.md              # 项目文档

---

## 🚀 快速开始

### 方式一：本地运行

**1. 安装依赖**

pip install -r requirements.txt

2. 启动 Ollama 服务

ollama serve

3. 下载模型（首次运行）

ollama pull deepseek-r1:8b

4. 启动 API 服务

uvicorn rag_api:app --reload

5. 启动前端界面（新开终端）

streamlit run app.py

6. 访问

· API 文档：http://localhost:8000/docs
· 前端界面：http://localhost:8501

---
方式二：Docker 部署

# 构建镜像
docker build -t rag-qa-system .

# 运行容器
docker run -p 8000:8000 rag-qa-system

---

🔄 RAG 完整流程

```
用户上传文档
    ↓
文档解析（TXT / PDF）
    ↓
文本分块（chunk_size=500, overlap=50）
    ↓
向量检索（语义匹配） + BM25 检索（关键词匹配）
    ↓
混合检索结果合并去重
    ↓
本地大模型生成回答（DeepSeek 8B）
    ↓
返回答案 + 引用原文出处
```

---

📊 功能完成度

功能模块 状态 说明
本地大模型部署 ✅ 完成 Ollama + DeepSeek 8B
文档解析 ✅ 完成 支持 TXT / PDF
文本分块 ✅ 完成 chunk_size=500, overlap=50
向量检索 ✅ 完成 Chroma + bge-small-zh
关键词检索 ✅ 完成 BM25 算法
混合检索 ✅ 完成 向量 + 关键词双路召回
问答溯源 ✅ 完成 显示引用原文
FastAPI 接口 ✅ 完成 RESTful API + Swagger
Streamlit 界面 ✅ 完成 可视化交互
Docker 部署 ✅ 完成 一键启动

---

🎯 与 v1 版本对比

维度 v1（云端 API 版） v2（本地私有化版）
大模型 DeepSeek API DeepSeek 8B（本地）
网络依赖 需要联网 完全离线
检索方式 纯向量检索 混合检索（向量+关键词）
问答溯源 无 有
前端界面 Swagger Swagger + Streamlit
部署方式 手动启动 Docker 一键部署
适用场景 快速验证 企业私有化生产

---

🗂️ 版本历史

· v2（当前）：本地模型 + 混合检索 + 问答溯源 + Docker 部署
· v1：云端 API + 向量检索

---

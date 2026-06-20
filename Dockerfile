FROM python:3.12-slim

WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制代码
COPY rag_api.py .
COPY bge-small-zh-v1.5 ./bge-small-zh-v1.5

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8000"]
import streamlit as st
import requests
import json

st.set_page_config(page_title="本地知识库问答", layout="wide")

st.title("📚 本地知识库问答系统")
st.markdown("基于 DeepSeek 8B 本地模型 + RAG")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 说明")
    st.markdown("""
    1. 上传文档（支持 TXT / PDF）
    2. 输入问题
    3. 系统会基于文档内容回答
    4. 显示引用原文出处
    """)
    st.divider()
    st.caption(f"API地址: http://localhost:8000")

# 主界面
uploaded_file = st.file_uploader("📄 上传文档", type=["txt", "pdf"])
question = st.text_input("💬 输入你的问题", placeholder="例如：这份文档主要讲了什么？")

if st.button("🚀 开始提问", type="primary"):
    if not uploaded_file:
        st.warning("请先上传文档")
    elif not question:
        st.warning("请输入问题")
    else:
        with st.spinner("正在处理...（本地模型推理中）"):
            try:
                files = {"file": uploaded_file}
                params = {"question": question}
                response = requests.post(
                    "http://localhost:8000/ask_with_file",
                    files=files,
                    params=params,
                    timeout=180
                )

                if response.status_code == 200:
                    data = response.json()

                    # 显示答案
                    st.success("✅ 回答成功")
                    st.subheader("📝 答案")
                    st.write(data["answer"])

                    # 显示来源
                    if data.get("sources"):
                        st.subheader("📖 引用原文出处")
                        for i, source in enumerate(data["sources"], 1):
                            with st.expander(f"来源 {i}"):
                                st.text(source[:500] + "..." if len(source) > 500 else source)
                else:
                    st.error(f"请求失败: {response.status_code}")

            except requests.exceptions.Timeout:
                st.error("⏰ 请求超时，模型可能还在加载，请稍后再试")
            except Exception as e:
                st.error(f"❌ 发生错误: {str(e)}")
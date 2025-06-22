# model3_v3.py
import streamlit as st
import torch
import threading
import datetime
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

NEW_MAX_TOKENS = 3072

# ------------------------- 模型加载（优化后分离） -------------------------
@st.cache_resource
def load_tokenizer():
    model_path = "/home/whu/qwen3/model"
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

@st.cache_resource
def load_model():
    model_path = "/home/whu/qwen3/model"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    if torch.cuda.is_available():
        model.half()
        torch.backends.cudnn.benchmark = True
    return model

tokenizer = load_tokenizer()
model = load_model()

# ------------------------- 通用流式生成 -------------------------
def stream_generate_response(prompt, enable_thinking, max_new_tokens=NEW_MAX_TOKENS):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        streamer=streamer,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for new_text in streamer:
        yield new_text

def run_stream_analysis(func, code, enable_thinking):
    result = ""
    for chunk in func(code, enable_thinking):
        result += chunk
    return result

# ------------------------- 三种分析方法 -------------------------
def explain_cpp_block(code,enable_thinking):
    prompt = f"""任务: 请你扮演一个经验丰富的C语言/c++/cuda代码分析师。你的任务是对我接下来提供的C代码进行详细分析，识别其中的关键元素，并将代码逻辑上分块。
具体步骤:
1.代码分析: 仔细阅读我提供的代码。识别出代码中的主要组成部分，例如：
- 头文件包含 (#include): 列出所有包含的头文件及其作用（如果可以判断）。
- 宏定义 (#define): 列出所有宏定义及其用途。
- 类型定义 (typedef): 如果有，列出自定义的类型及其含义。
- 全局变量/常量: 识别出在函数外部定义的变量或使用 const/static 修饰的变量。
- 函数定义: 识别出所有自定义函数，包括其返回类型、参数列表和函数体。
- 结构体/联合体/枚举 (struct/union/enum): 识别出所有自定义的数据结构及其成员。
- 主函数 (main): 特别关注程序的入口点。
- 注释: 注意代码中的注释，它们有助于理解代码意图。
- 其他逻辑块: 识别出代码中其他有意义的逻辑片段，例如特定的代码段、循环、条件判断等。
- 逻辑分块: 基于第一步的分析，将代码按照其逻辑功能或结构进行划分。每个分块应该代表一个相对独立或功能明确的代码单元。例如，一个函数可以是一个分块，一个结构体定义可以是一个分块，处理特定任务的一段代码也可以是一个分块。
2.列表呈现: 将上述分块以列表的形式呈现给我。列表的每个条目应包含以下信息：
3.分块名称/标识: 用简洁明了的词语命名这个分块，例如“函数：calculateSum”、“结构体定义：Person”、“主函数入口”、“全局变量声明”等。
4.解释每一个代码块的功能与作用！！
5.代码中不存在的关键元素，禁止输出！！
输入:
{code}
"""
    return stream_generate_response(prompt,enable_thinking)

def analyze_performance(code,enable_thinking):
    prompt = f"""请从下面这段代码中分析其性能问题，包括：
- 时间复杂度和空间复杂度
- 是否有不必要的内存申请或释放：堆、栈、常量区、寄存器等
- 是否存在循环优化空间
- STL 使用是否合理
- 指针/智能指针使用及其代码实现
- 多线程或异步潜力
- 测试程序性能及其代码
- 总体性能瓶颈点
```cpp\n{code}\n```"""
    return stream_generate_response(prompt,enable_thinking)

# 优化建议
def analyze_optimization(code,enable_thinking):
    prompt = f"""请总结以下代码的功能与结构，并提出可行的优化建议：
```cpp\n{code}\n```"""
    return stream_generate_response(prompt,enable_thinking)

# UI 构建
st.set_page_config(page_title="程序编写、分析与优化", layout="wide")

# 页面头部信息（居中标题 + 作者 + 邮箱）
st.markdown("""
<div style="text-align: center;">
    <h1>📘 程序编写、分析与优化</h1>
    <p><strong>👤 作者：</strong>jianbang zhang</p>
    <p><strong>📧 联系邮箱：</strong><a href="mailto:queju@example.com">whdx072018@foxmail.com</a></p>
</div>
""", unsafe_allow_html=True)

st.set_page_config(page_title="代码结构分块与解释", layout="wide")
code = st.text_area("请输入代码或任务：", height=300)
mode = st.radio("选择分析模式：", ["⚡ 快速分析", "🌊 深度分析"])
enable_thinking = mode.startswith("🌊")

if st.button("🧹 清理模型缓存"):
    load_model.clear()
    load_tokenizer.clear()
    st.success("✅ 模型缓存已清除。下次分析将重新加载模型。")

if st.button("🚀 开始分块分析"):
    if not code.strip():
        st.warning("请输入有效的代码或任务：")
    else:
        all_results = f"# ⏱️ 分析时间：{datetime.datetime.now()}\n\n"
        all_results += f"## 💻 输入代码\n```cpp\n{code}\n```\n\n"

        # ✅ 第1部分：结构分块分析（流式）
        st.markdown("## 🧱 分块结构解释")
        with st.expander("🔍 点击展开结构分析", expanded=True):
            explanation = ""
            explain_area = st.empty()
            for chunk in explain_cpp_block(code, enable_thinking):
                explanation += chunk
                explain_area.markdown(explanation)
            all_results += f"## 🔍 分块解释\n{explanation}\n\n"

        # ✅ 第2部分：性能分析 + 优化建议（并发加速）
        with st.spinner("正在分析性能与优化建议..."):
            with ThreadPoolExecutor() as executor:
                perf_future = executor.submit(run_stream_analysis, analyze_performance, code, enable_thinking)
                opt_future = executor.submit(run_stream_analysis, analyze_optimization, code, enable_thinking)

                perf_result = perf_future.result()
                opt_result = opt_future.result()

        st.markdown("## 📈 性能分析")
        with st.expander("💬 点击展开性能分析", expanded=True):
            st.markdown(perf_result)
        all_results += f"## 📈 性能分析\n{perf_result}\n\n"

        st.markdown("## 🛠 优化建议")
        with st.expander("💬 点击展开优化建议", expanded=True):
            st.markdown(opt_result)
        all_results += f"## 🛠 优化建议\n{opt_result}\n\n"

        # ✅ Markdown 导出
        filename = f"code_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        st.download_button(
            label="💾 保存分析结果为 Markdown 文件",
            data=all_results,
            file_name=filename,
            mime="text/markdown"
        )


st.markdown("---")
st.markdown("""
### 🔧 功能说明
- ✅ 自动结构分析 + 性能瓶颈分析 + 优化建议
- ✅ 流式逐块输出 + 一键保存为 Markdown
- ✅ 分析模式切换 + 缓存清理按钮
- ✅ 分析过程多线程并发，加速运行
""")

# model3_v3.py
import streamlit as st
import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

NEW_MAX_TOKENS = 3072
# 模型加载
@st.cache_resource
def load_model():
    model_path = "/home/whu/qwen3/model"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
    return model, tokenizer

model, tokenizer = load_model()

# 流式生成
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

# 按结构模块分块提取代码
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

def analyze_performance(code):
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
def analyze_optimization(code):
    prompt = f"""请总结以下代码的功能与结构，并提出可行的优化建议：
```cpp\n{code}\n```"""
    return stream_generate_response(prompt,enable_thinking)

# UI 构建
st.title("程序分析与优化")

st.set_page_config(page_title="代码结构分块与解释", layout="wide")
code = st.text_area("请输入代码：", height=300)

mode = st.radio("选择分析模式：", ["⚡ 快速分析", "🌊 深度分析"])
enable_thinking = mode.startswith("🌊")

if st.button("🚀 开始分块分析"):
    if not code.strip():
        st.warning("请输入有效的 C++ 代码")
    else:
        st.info("📦 正在分块提取结构并分析模块...")
        st.markdown("## 🔍 分块解释")
        with st.expander("💬 点击展开解释", expanded=True):
            explanation = ""
            explain_area = st.empty()
            for chunk in explain_cpp_block(code, enable_thinking):
                explanation += chunk
                explain_area.markdown(explanation)
        st.markdown("---")

        st.markdown("## 📈 性能分析")
        perf_result = ""
        perf_area = st.empty()
        for chunk in analyze_performance(code):
            perf_result += chunk
            perf_area.markdown(perf_result)

        st.markdown("## 🛠 优化建议")
        opt_result = ""
        opt_area = st.empty()
        for chunk in analyze_optimization(code):
            opt_result += chunk
            opt_area.markdown(opt_result)

st.markdown("---")
st.markdown("""
### 🔧 功能说明
- ✅ 自动将 C++ 代码按结构划分为多个模块（类/函数/变量）
- ✅ 每块结构自动识别并生成解释
- ✅ 流式逐句输出解释内容
- ✅ 不使用 JSON，保持自然代码分块风格
""")


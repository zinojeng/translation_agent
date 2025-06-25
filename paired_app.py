import streamlit as st
import streamlit.components.v1 as components
import os
from litellm import completion
import PyPDF2
from docx import Document
import tiktoken
import nltk
import ssl
import diff_match_patch as dmp_module
from typing import List, Tuple
import requests
from datetime import datetime
import time
import threading
from streamlit.runtime.scriptrunner import get_script_run_ctx

# SSL and NLTK setup
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt', quiet=True)
# Set page config
st.set_page_config(page_title="Translation Agent", layout="wide")
# Force redeploy

#Add custom CSS to hide the GitHub icon and improve mobile sidebar
hide_github_icon = """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, 
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, 
    .viewerBadge_text__1JaDK{ display: none; } 
    #MainMenu{ visibility: hidden; } 
    footer { visibility: hidden; } 
    header { visibility: hidden; }
    
    /* Mobile sidebar improvements */
    @media (max-width: 768px) {
        .css-1d391kg { 
            padding-top: 1rem; 
        }
        
        /* Make sidebar toggle button more prominent */
        .css-1rs6os { 
            display: block !important;
            position: sticky;
            top: 0;
            z-index: 999;
        }
    }
    
    /* Sidebar toggle button styling */
    .sidebar-toggle {
        position: fixed;
        top: 10px;
        left: 10px;
        z-index: 1000;
        background: #ff4b4b;
        color: white;
        border: none;
        padding: 8px 12px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
    }
    </style>
"""
st.write(hide_github_icon, unsafe_allow_html=True)


# 常量定义
USD_TO_NTD_RATE = 30

# Sidebar for model selection
model_options = {
    "gpt-4o-mini": {"input_cost": 0.15, "output_cost": 0.60, "model_name": "gpt-4o-mini"},
    "gpt-4o": {"input_cost": 2.50, "output_cost": 10.00, "model_name": "gpt-4o"},
    "o1-mini": {"input_cost": 1.10, "output_cost": 4.40, "model_name": "o1-mini"},
    "o3-mini": {"input_cost": 1.10, "output_cost": 4.40, "model_name": "o3-mini"},
    "deepseek-chat": {
        "input_cost": 0.015, 
        "output_cost": 0.06,
        "model_name": "deepseek-chat"
    },
    "deepseek-reasoner": {
        "input_cost": 0.03, 
        "output_cost": 0.12,
        "model_name": "deepseek-reasoner"
    }
}

# Configuration function
def show_configuration(container_type="sidebar"):
    if container_type == "sidebar":
        container = st.sidebar
        container.title("Configuration")
    else:
        container = st
        container.subheader("📋 模型和 API 設定")
    
    # Model selection with o3-mini as default
    selected_model = container.selectbox(
        "Select Translation Model:",
        list(model_options.keys()),
        index=list(model_options.keys()).index("o3-mini"),  # Set o3-mini as default
        help=(
            "Batch input/output cost per 1M tokens (USD):\n\n"
            "gpt-4o: 2.50/10.00, gpt-4o-mini: 0.15/0.60\n"
            "o1-mini: 1.10/4.40, o3-mini: 1.10/4.40\n\n"
            "deepseek-chat: 0.015/0.06, deepseek-reasoner: 0.03/0.12\n\n"
            "For OpenAI model pricing details, visit: https://platform.openai.com/docs/pricing\n\n"
            "For DeepSeek model pricing details, visit: https://api-docs.deepseek.com/quick_start/pricing"
        ),
        key=f"model_selection_{container_type}"
    )
    
    # Initialize API key variables
    openai_api_key = None
    deepseek_api_key = None
    
    # API key input based on selected model
    if selected_model.startswith("deepseek"):
        deepseek_api_key = container.text_input(
            label="DeepSeek API Key:",
            type='password',
            placeholder="sk-...",
            help="Get from https://platform.deepseek.com/api_keys",
            key=f"deepseek_api_key_{container_type}"
        )
        if deepseek_api_key:
            os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
    else:
        openai_api_key = container.text_input(
            label="OpenAI API Key:",
            type='password',
            placeholder="sk-...",
            help="Get from https://platform.openai.com/account/api-keys",
            key=f"openai_api_key_{container_type}"
        )
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
    
    return selected_model, openai_api_key, deepseek_api_key

# Show configuration in sidebar
selected_model, openai_api_key, deepseek_api_key = show_configuration("sidebar")

# Set MODEL_NAME after selection
MODEL_NAME = selected_model

# Set costs after model selection
INPUT_COST_PER_1K_TOKENS = model_options[MODEL_NAME]["input_cost"]
OUTPUT_COST_PER_1K_TOKENS = model_options[MODEL_NAME]["output_cost"]

# Calculate prices
input_cost_ntd = INPUT_COST_PER_1K_TOKENS * USD_TO_NTD_RATE
output_cost_ntd = OUTPUT_COST_PER_1K_TOKENS * USD_TO_NTD_RATE


st.sidebar.markdown("""
**Modified by:** Tseng Yao Hsien \n
**Contact:** zinojeng@gmail.com  \n
**Reference:** Andrew Ng's AI Agent System for Language Translation [https://github.com/andrewyng/translation-agent](https://github.com/andrewyng/translation-agent)

**Translation Agent** 旨在提升機器翻譯的品質。它採用獨特的「省思式工作流程」，模擬人類翻譯專家的思考過程：

1.**初始翻譯**: 利用大型語言模型 (LLM) 產生初步譯文。\n
2.**反思與改進**: 引導 LLM 反思自身譯文，提出改進建議，如同語言審核者般找出不足之處。\n
3.**優化輸出**: 根據 LLM 的建議，或醫療次專科角色，優化譯文，使其更精確、流暢，並符合目標語言的慣用表達。\n
""")

    
# Language selection
st.title("Translation Agent: Agentic translation using reflection workflow")

# Mobile sidebar access reminder and toggle button
col_btn, col_info = st.columns([1, 4])

with col_btn:
    if st.button("📋 設定", key="sidebar_toggle", help="點擊打開側邊欄設定"):
        st.session_state.show_sidebar_content = not st.session_state.get('show_sidebar_content', False)

with col_info:
    st.info("📱 手機用戶：點擊左側「📋 設定」按鈕來顯示模型和 API 設定選項。")

# Show configuration in main area if toggled (for mobile users)
if st.session_state.get('show_sidebar_content', False):
    with st.expander("🔧 模型和 API 設定", expanded=True):
        main_selected_model, main_openai_api_key, main_deepseek_api_key = show_configuration("main")
        # Update global variables if main configuration is used
        if main_selected_model:
            MODEL_NAME = main_selected_model
            if main_openai_api_key:
                openai_api_key = main_openai_api_key
            if main_deepseek_api_key:
                deepseek_api_key = main_deepseek_api_key

st.subheader("Select Languages")
col1, col2, col3 = st.columns(3)

with col1:
    source_lang = st.selectbox(
        "Source Language",
        ["Chinese", "English", "Spanish", "French", "German", "Italian", "Japanese", "Korean", "Vietnamese", "Indonesian", "Thai"]
    )

with col2:
    target_lang = st.selectbox(
        "Target Language",
        ["English", "Traditional Chinese", "Simplified Chinese", "Spanish", "French", "German", "Italian", "Japanese", "Korean", "Vietnamese", "Indonesian", "Thai"]
    )

with col3:
    country_options = {
        "Traditional Chinese": ["Taiwan", "Hong Kong"],
        "Simplified Chinese": ["China", "Singapore"],
        "English": ["USA", "UK", "Australia", "Canada", "Philippines"],
        "Spanish": ["Spain", "Mexico", "Argentina"],
        "French": ["France", "Canada", "Belgium"],
        "German": ["Germany", "Austria", "Switzerland"],
        "Italian": ["Italy", "Switzerland"],
        "Japanese": ["Japan"],
        "Korean": ["South Korea"],
        "Vietnamese": ["Vietnam"],
        "Indonesian": ["Indonesia"],
        "Thai": ["Thailand"]
    }
    country = st.selectbox("Country/Region", country_options.get(target_lang, []))


# 添加翻譯模式選項
st.subheader("Select Translation Mode")

translation_modes = [
    "Standard", "Fluency", "Natural", "Formal", "Academic",
    "Simple", "Creative", "Expand", "Shorten"
]

selected_translation_mode = st.selectbox(
    "Select Translation Mode:",
    translation_modes,
    help="Choose the translation mode that best suits your needs."
)

# 設置不同模式的 prompts 和 temperature
prompts = {
    "Standard": {"prompt": "Please provide a translation of the following text with standard fluency and accuracy, maintaining the original meaning: ", "temperature": 0.5},
    "Fluency": {"prompt": "Please translate the following text, ensuring it is highly fluent, coherent, and easy to read, while preserving the original intent: ", "temperature": 0.7},
    "Natural": {"prompt": "Please translate the following text in a natural, conversational tone that sounds like everyday speech: ", "temperature": 0.9},
    "Formal": {"prompt": "Please translate the following text in a formal and professional manner, suitable for business or official contexts: ", "temperature": 0.3},
    "Academic": {"prompt": "Please translate the following text in an academic and scholarly style, appropriate for research papers or academic articles: ", "temperature": 0.2},
    "Simple": {"prompt": "Please translate the following text using simple, clear, and straightforward language that is easy to understand: ", "temperature": 0.5},
    "Creative": {"prompt": "Please translate the following text with a creative and engaging style, adding flair and originality while maintaining the core message: ", "temperature": 1.0},
    "Expand": {"prompt": "Please translate the following text, expanding on ideas where necessary to provide additional context and detail: ", "temperature": 0.8},
    "Shorten": {"prompt": "Please translate the following text, shortening the content where possible while retaining the essential information: ", "temperature": 0.5}
}

# 根據選擇的模式設置 prompt 和 temperature
selected_prompt_template = prompts[selected_translation_mode]["prompt"]
selected_temperature = prompts[selected_translation_mode]["temperature"]


# 專科介紹
specialties = {
    "內科": [
        "內分泌暨新陳代謝科",
        "胸腔內科",
        "職業醫學科",
        "一般內科",
        "神經內科",
        "腎臟內科",
        "心臟內科",
        "過敏免疫風濕科",
        "感染科",
        "家庭醫學科",
        "胃腸肝膽科",
        "血液腫瘤科",
    ],
    "外科": [
        "骨科",
        "一般外科",
        "神經外科",
        "泌尿科",
        "胸腔外科",
        "心臟血管外科",
        "整形外科",
        "大腸直腸外科",
        "小兒外科",
    ],
    "其他專科": [
        "婦產部",
        "兒童醫學部",
        "眼科",
        "核子醫學科",
        "麻醉科",
        "耳鼻喉部",
        "皮膚科",
        "心身科",
        "放射腫瘤科",
        "放射診斷科",
        "復健醫學部",
        "急診醫學部",
        "口腔醫學部",
        "中醫科",
    ],
}

# 專科選擇
st.write("Select a Medical Specialty (Optional)")
selected_department = st.selectbox("選擇科別 (可選)", ["無"] + list(specialties.keys()))

if selected_department != "無":
    selected_specialty = st.selectbox("選擇專科 (可選)", ["無"] + specialties[selected_department])
else:
    selected_specialty = "無"

# Input method selection
input_method = st.radio("Choose input method:", ("Enter Text", "Upload PDF", "Upload TXT", "Upload Word Document"))

# Function to read PDF
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
# Function to read TXT
def read_txt(file):
    return file.getvalue().decode("utf-8")

# Function to read Word Document
def read_doc_or_docx(file):
    file_extension = file.name.split('.')[-1].lower()
    try:
        if file_extension == 'docx':
            doc = Document(file)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return '\n'.join(full_text)
        elif file_extension == 'doc':
            st.error("Legacy .doc files are not supported. Please convert to .docx format.")
            return ""
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return ""
    
# Input text based on selected method
if input_method == "Upload PDF":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        source_text = read_pdf(uploaded_file)
        st.text_area("Extracted text from PDF:", value=source_text, height=200)
    else:
        source_text = ""
elif input_method == "Upload TXT":
    uploaded_file = st.file_uploader("Choose a TXT file", type="txt")
    if uploaded_file is not None:
        source_text = read_txt(uploaded_file)
        st.text_area("Extracted text from TXT:", value=source_text, height=200)
    else:
        source_text = ""
elif input_method == "Upload Word Document":
    uploaded_file = st.file_uploader("Choose a Word Document", type=["docx"])
    if uploaded_file is not None:
        source_text = read_doc_or_docx(uploaded_file)
        if source_text:
            st.text_area("Extracted text from Word Document:", value=source_text, height=200)
        else:
            st.error("Failed to extract text from the document.")
    else:
        source_text = ""
else:  # Enter Text
    source_text = st.text_area("Enter the text to translate:", height=200)


def estimate_token_count(text):
    try:
        model_name = model_options[MODEL_NAME]["model_name"]
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback to a default encoding for models not supported by tiktoken
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))


def estimate_cost(input_tokens, output_tokens):
    input_cost = (input_tokens / 1_000_000) * model_options[MODEL_NAME]["input_cost"]
    output_cost = (output_tokens / 1_000_000) * model_options[MODEL_NAME]["output_cost"]
    total_cost_usd = input_cost + output_cost
    return total_cost_usd * USD_TO_NTD_RATE


# Translation functions
def get_completion(user_prompt, system_message="You are a helpful assistant.", model=MODEL_NAME, temperature=0.2):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
    
    # 根據模型選擇不同的 API endpoint 和設定
    model_name = model_options[model]["model_name"]
    
    try:
        if model.startswith("deepseek"):
            if not deepseek_api_key:
                raise ValueError("DeepSeek API key is required for DeepSeek models")
            
            # 使用 OpenAI SDK 調用 DeepSeek API
            from openai import OpenAI
            import httpx
            
            # 創建自定義的 httpx client，不使用代理
            http_client = httpx.Client()
            
            client = OpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com/v1",
                http_client=http_client
            )
            
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=min(temperature, 1.3)  # DeepSeek 最大支援 1.3
                )
                
                return response.choices[0].message.content
            finally:
                http_client.close()
            
        else:
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI models")
            os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
            os.environ["OPENAI_API_KEY"] = openai_api_key
            
            # 如果是 O-series 模型，強制使用 temperature=1
            if model.startswith("o"):
                temperature = 1
            
            response = completion(
                model=model_name,
                messages=messages, 
                temperature=temperature
            )
            return response["choices"][0]["message"]["content"]
            
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        raise e

def one_chunk_initial_translation(model, source_text):
    system_message = f"You are an expert medical translator, specializing in translating medical instructions and educational materials from {source_lang} to {target_lang}."
    translation_prompt = f"""This is a translation from {source_lang} to {target_lang}. {selected_prompt_template} \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_text}

{target_lang}:"""
    return get_completion(translation_prompt, system_message=system_message, model=model, temperature=selected_temperature)


# def one_chunk_initial_translation_original(model, source_text):
#     system_message = f"You are an expert medical translator, specializing in translating medical instructions and educational materials from {source_lang} to {target_lang}."
#     translation_prompt = f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
# Do not provide any explanations or text apart from the translation.
# {source_lang}: {source_text}

# {target_lang}:"""
#     return get_completion(translation_prompt, system_message=system_message, model=model)

def one_chunk_reflect_on_translation(model, source_text, translation_1):
    system_message = f"{selected_prompt_template} \
    You are an expert medical translator specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."
    prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""
    return get_completion(prompt, system_message=system_message, model=model, temperature=selected_temperature)

def one_chunk_improve_translation(model, source_text, translation_1, reflection):
    system_message = f"{selected_prompt_template} \
    You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."
    prompt = f"""Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang}, taking into
account a list of expert suggestions and constructive criticisms.

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Provide your improved translation as a continuous text, without any additional formatting or labels."""
        # Add specialty information (if a specialty is selected)
    if selected_specialty != "無":
        prompt += f"This translation is for medical instructions related to {selected_specialty}. Please ensure the terminology is accurate and appropriate for this specialty.\n\n"

    # Complete the prompt
    prompt += """Provide your improved translation as a continuous text, without any additional formatting or labels."""

    return get_completion(prompt, system_message, model=model, temperature=selected_temperature)

def one_chunk_translate_text(model, source_text):
    try:
        # 初始翻译
        translation_1 = one_chunk_initial_translation(model, source_text)
        # 反思翻译
        reflection = one_chunk_reflect_on_translation(model, source_text, translation_1)
        # 改进翻译
        improved_translation = one_chunk_improve_translation(model, source_text, translation_1, reflection)

        return {
            "initial_translation": translation_1,
            "reflection": reflection,
            "improved_translation": improved_translation,
        }
    except Exception as e:
        st.error(f"An error occurred during translation processing: {str(e)}")
        return None


#Compared text difference 
def compare_texts(source_text, improved_translation):
    # 創建 diff_match_patch 物件
    dmp = dmp_module.diff_match_patch()

    # 生成差異
    diff = dmp.diff_main(source_text, improved_translation)
    dmp.diff_cleanupSemantic(diff)

    # 輸出差異為 HTML
    html_diff = dmp.diff_prettyHtml(diff)

    # 在 Streamlit 中顯示 HTML 格式的比較結果
    st.markdown(html_diff, unsafe_allow_html=True)


 # 修改 perform_translation 函数
def perform_translation():
    # 檢查 API key
    selected_provider = "openai" if not MODEL_NAME.startswith("deepseek") else "deepseek"
    
    if selected_provider == "openai" and not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        return
    elif selected_provider == "deepseek" and not deepseek_api_key:
        st.error("Please enter your DeepSeek API key in the sidebar.")
        return
    
    if not source_text:
        st.error("Please provide some text to translate.")
        return
    
    try:
        with st.spinner("Translating... This may take a moment."):
            # 执行完整文本的翻译
            full_translation_result = one_chunk_translate_text(MODEL_NAME, source_text)
            
            if full_translation_result is None:
                st.error("Full text translation failed. Please try again.")
                return
            
            # 显示完整文本的翻译结果
            st.subheader("Full Text Translation")
            st.markdown("**Initial Translation:**")
            st.write(full_translation_result["initial_translation"])
            
            st.markdown("**Translation Reflection:**")
            st.write(full_translation_result["reflection"])
            
            # st.markdown("**Improved Translation:**")
            # st.write(full_translation_result["improved_translation"])

            # 呼叫 compare_texts 函數進行比較
            st.markdown("**Compare text differences:**")
            compare_texts(full_translation_result["initial_translation"], full_translation_result["improved_translation"])

            # 顯示完整文本的翻譯結果
            st.write("**Final Translation Results**")
            st.markdown(f"**原文:**\n{source_text}")
            st.markdown(f"**譯文:**\n{full_translation_result['improved_translation']}")

            # 计算 token 使用量和估算成本
            input_tokens = estimate_token_count(source_text)
            output_tokens = (estimate_token_count(full_translation_result['initial_translation']) + 
                             estimate_token_count(full_translation_result['reflection']) + 
                             estimate_token_count(full_translation_result['improved_translation']))
            total_tokens = input_tokens + output_tokens
            estimated_cost = estimate_cost(input_tokens, output_tokens)

            st.subheader("Token Usage and Cost Estimation")
            st.write(f"Total tokens used: {total_tokens}")
            st.write(f"Estimated cost: NTD {estimated_cost:.3f}")
            total_characters = len(source_text)
            st.write(f"Total characters in source text: {total_characters}")
            st.write(f"Model: {MODEL_NAME}: US{INPUT_COST_PER_1K_TOKENS}/US{OUTPUT_COST_PER_1K_TOKENS} per 1K tokens for batch API input/output, respectively")

        st.success("Translation completed!")
        
        # 准备下载按钮
        result_text = "Source Text, Improved Translations:\n\n"
        result_text += f"原文:\n{source_text}\n\n"
        result_text += f"改善後譯文 (Full Text):\n{full_translation_result['improved_translation']}\n\n"
        # result_text += "Sentence-by-Sentence Translation:\n\n"

        result_text += f"\nEstimated Cost: NTD {estimated_cost:.3f}\n"
        result_text += f"Total characters in source text: {total_characters}\n"

        st.download_button(
            label="Download Translation Results",
            data=result_text,
            file_name="translation_results.txt",
            mime="text/plain"
        )
    except Exception as e:
        st.error(f"An error occurred during translation: {str(e)}")
        st.exception(e)  
        

# 主函数
def main():
    if st.button("Translate"):
        perform_translation()
        st.info("Execution finished")

# 添加心跳檢測路由
@st.cache_data(ttl=60)
def heartbeat():
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat()
    }

# 添加自我喚醒功能
def keep_alive():
    while True:
        try:
            requests.get("https://translationagent.streamlit.app")
            if 'last_ping' in st.session_state:
                st.session_state['last_ping'] = time.time()
            time.sleep(86400)  # 每24小時喚醒一次
        except Exception as e:
            print(f"Keep-alive error: {e}")
            time.sleep(60)  # 如果出錯，1分鐘後重試

# 在主程式開始時啟動自我喚醒線程
if 'keep_alive_thread' not in st.session_state:
    keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
    keep_alive_thread.start()
    st.session_state['keep_alive_thread'] = keep_alive_thread

# 顯示最後喚醒時間（可選）
if 'last_ping' in st.session_state:
    st.sidebar.text(f"上次活動時間: {time.ctime(st.session_state['last_ping'])}")

if __name__ == "__main__":
    # 使用 query_params 替代 experimental_get_query_params
    if "heartbeat" in st.query_params:
        st.write(heartbeat())
    else:
        main()


#在最後嵌入 JavaScript SDK 代碼
components.html(
    """
    <script src="https://sf-cdn.coze.com/obj/unpkg-va/flow-platform/chat-app-sdk/0.1.0-beta.5/libs/oversea/index.js"></script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            // CozeWebSDK initialization
            new CozeWebSDK.WebChatClient({
                config: {
                    bot_id: '7408278430665752583',
                },
                componentProps: {
                    title: 'Gestational Diabetes Care',
                    icon: 'https://www.icareweight.com/wp-content/uploads/2024/08/images-5-removebg-preview.png',
                },
            });
            
            // Mobile sidebar toggle enhancement
            function enhanceMobileSidebar() {
                // Add a visible sidebar toggle button for mobile
                const sidebarToggle = document.querySelector('[data-testid="collapsedControl"]');
                if (sidebarToggle) {
                    sidebarToggle.style.backgroundColor = '#ff4b4b';
                    sidebarToggle.style.color = 'white';
                    sidebarToggle.style.padding = '8px';
                    sidebarToggle.style.borderRadius = '4px';
                    sidebarToggle.style.position = 'fixed';
                    sidebarToggle.style.top = '10px';
                    sidebarToggle.style.left = '10px';
                    sidebarToggle.style.zIndex = '1000';
                    sidebarToggle.title = '點擊打開側邊欄設定';
                }
            }
            
            // Check periodically for sidebar toggle button
            setInterval(enhanceMobileSidebar, 1000);
        });
    </script>
    """,
    height=600
)


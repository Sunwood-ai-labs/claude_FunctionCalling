import gradio as gr
from anthropic import Anthropic
import re
import os

# python-dotenvをインポート
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

# APIキーを環境変数から取得
api_key = os.getenv("ANTHROPIC_API_KEY")

client = Anthropic(api_key=api_key)
MODEL_NAME = "claude-3-haiku-20240307"

def construct_format_tool_for_claude_prompt(name, description, parameters):
    constructed_prompt = (
        "<tool_description>\n"
        f"<tool_name>{name}</tool_name>\n"
        "<description>\n"
        f"{description}\n"
        "</description>\n"
        "<parameters>\n"
        f"{construct_format_parameters_prompt(parameters)}\n"
        "</parameters>\n"
        "</tool_description>"
    )
    return constructed_prompt

def construct_format_parameters_prompt(parameters):
    constructed_prompt = "\n".join(
        f"<parameter>\n<name>{parameter['name']}</name>\n<type>{parameter['type']}</type>\n<description>{parameter['description']}</description>\n</parameter>"
        for parameter in parameters
    )
    return constructed_prompt

def construct_tool_use_system_prompt(tools):
    tool_use_system_prompt = (
        "この環境では、ユーザーの質問に答えるための一連のツールにアクセスできます。\n"
        "\n"
        "次のように呼び出すことができます。\n"
        "<function_calls>\n"
        "<invoke>\n"
        "<tool_name>$TOOL_NAME</tool_name>\n"
        "<parameters>\n"
        "<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>\n"
        "...\n"
        "</parameters>\n"
        "</invoke>\n"
        "</function_calls>\n"
        "\n"
        "利用可能なツールは次のとおりです。\n"
        "<tools>\n" +
        '\n'.join([tool for tool in tools]) +
        "\n</tools>"
    )
    return tool_use_system_prompt

def extract_between_tags(tag: str, string: str, strip: bool = False) -> list[str]:
    ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
    if strip:
        ext_list = [e.strip() for e in ext_list]
    return ext_list

def construct_successful_function_run_injection_prompt(invoke_results):
    constructed_prompt = (
        "<function_results>\n"
        + '\n'.join(
            f"<result>\n<tool_name>{res['tool_name']}</tool_name>\n<stdout>\n{res['tool_result']}\n</stdout>\n</result>"
            for res in invoke_results
        ) + "\n</function_results>"
    )
    return constructed_prompt

def create_function(name, description, parameters, python_function):
    tool = construct_format_tool_for_claude_prompt(name, description, parameters)
    system_prompt = construct_tool_use_system_prompt([tool])
    
    def chat(message):
        function_calling_message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            messages=[{"role": "user", "content": message}],
            system=system_prompt,
            stop_sequences=["\n\nHuman:", "\n\nAssistant:", "</function_calls>"]
        ).content[0].text
        
        params = {}
        for param in parameters:
            param_value = extract_between_tags(param["name"], function_calling_message)
            if param["type"] == "int":
                params[param["name"]] = int(param_value[0]) if param_value else 0
            else:
                params[param["name"]] = param_value[0] if param_value else ""

        result = python_function(**params)
        
        formatted_results = [{'tool_name': name, 'tool_result': result}]
        function_results = construct_successful_function_run_injection_prompt(formatted_results)
        
        partial_assistant_message = function_calling_message + "</function_calls>" + function_results
        
        final_message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": message},
                {"role": "assistant", "content": partial_assistant_message}
            ],
            system=system_prompt
        ).content[0].text
        
        return partial_assistant_message + final_message

    return chat

def calculator(num1, num2, operation):
    if operation == '+':
        return num1 + num2
    elif operation == "-":
        return num1 - num2
    elif operation == "*":
        return num1 * num2
    elif operation == "/":
        return num1 / num2
    else:
        return "エラー：サポートされていない演算です。"
        
calculator_params = [
    {
        "name": "first_operand",
        "type": "int",
        "description": "第1オペランド（演算子の前）"
    },
    {
        "name": "second_operand", 
        "type": "int",
        "description": "第2オペランド（演算子の後）"
    },
    {
        "name": "operator",
        "type": "str", 
        "description": "実行する演算。+、-、*、/のいずれかでなければなりません"
    }
]

calculator_func = create_function("calculator", "基本的な算術を行うための電卓関数です。加算、減算、乗算をサポートしています。", calculator_params, calculator)

def custom_function(name, description, parameters, python_function):
    return create_function(name, description, parameters, python_function)

def launch():
    with gr.Blocks() as demo:
        gr.Markdown("# Claude 関数呼び出しデモ")
        
        with gr.Tab("電卓"):
            calculator_input = gr.Textbox(label="質問を入力してください")
            calculator_output = gr.Textbox(label="Claude の回答")
            calculator_button = gr.Button("送信")
            calculator_button.click(calculator_func, inputs=calculator_input, outputs=calculator_output)

        with gr.Tab("カスタム関数"):
            func_name = gr.Textbox(label="関数名")
            func_description = gr.Textbox(label="関数の説明")
            func_params = gr.DataFrame(
                headers=["name", "type", "description"],
                datatype=["str", "str", "str"],
                label="関数のパラメータ",
                row_count=3
            )
            func_code = gr.Textbox(label="Pythonコード")
            func_create_button = gr.Button("関数を作成")
            
            func_input = gr.Textbox(label="質問を入力してください")
            func_output = gr.Textbox(label="Claude の回答")
            func_run_button = gr.Button("送信")
            
            def create_custom_func(name, description, params, code):
                params_dict = [{"name": row[0], "type": row[1], "description": row[2]} for row in params]
                exec(code, globals())
                custom_func = create_function(name, description, params_dict, eval(name))
                return custom_func
                
            func_create_button.click(create_custom_func, inputs=[func_name, func_description, func_params, func_code], outputs=[func_run_button.click])
            func_run_button.click(lambda x: x, inputs=func_input, outputs=func_output)

    demo.launch()

if __name__ == "__main__":
    launch()
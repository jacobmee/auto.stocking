import os
import json
from openai import OpenAI
import pandas as pd

# 调用 deepseek 函数
from Ashare import get_price
from openai import OpenAI
    
if __name__ == "__main__":


    # Ensure the API key is fetched from the correct environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("The OPENAI_API_KEY environment variable is not set. Please set it to proceed.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com")

    # Load prompt content from prompt.md
    prompt_file = "prompt.md"
    if os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as file:
            prompt_content = file.read()
    else:
        raise FileNotFoundError("The prompt.md file is not found. Please ensure it exists in the current directory.")

    # Fetch and include data for 沪深300指数
    hs300_df = get_price('000300.XSHG', frequency='60m', count=60)  # 支持'1d'日, '1w'周, '1M'月
    message = f"\n沪深300指数 小时线行情\n{hs300_df}"

    # Fetch and include data for 科创指数
    kc_df = get_price('000688.XSHG', frequency='60m', count=60)  # 支持'1m','5m','15m','30m','60m'
    message += f"\n科创指数 小时线行情\n{kc_df}"

    # Load trade operations from trade_ops.json
    trade_ops_file = "trade_ops.json"
    if os.path.exists(trade_ops_file):
        with open(trade_ops_file, "r", encoding="utf-8") as file:
            trade_ops_json = json.load(file)
            trade_ops_json_string = json.dumps(trade_ops_json, ensure_ascii=False, indent=4)
    else:
        trade_ops_json_string = "{}"  # Default to empty JSON if file does not exist

    message += f"\n你的交易记录在这里:\n{trade_ops_json_string}"
    message += f"\n现在告诉我你的买卖决策，按照这个json文件的格式给我输出"

    print("Sending prompt to Deepseek:")
    print(prompt_content)
    print("Sending message to Deepseek:")
    print(message) 
    
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": prompt_content},
            {"role": "user", "content": message},
        ],
        stream=False
    )
    
    print(response.choices[0].message.content)

    # Overwrite trade_today.json with response content
    try:
        with open('trade_today.json', 'w', encoding='utf-8') as f:
            response_content = response.choices[0].message.content
            # Assuming the response content is a valid JSON string
            json_data = json.loads(response_content)
            json.dump(json_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error overwriting trade_today.json with response content: {e}")


import os
from venv import logger
from openai import OpenAI
import yfinance as yf

# 调用 deepseek 函数
from openai import OpenAI

def ask_deepseek(stock_code, email_recipient):
    # Ensure the API key is fetched from the correct environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("The OPENAI_API_KEY environment variable is not set. Please set it to proceed.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com")

    # Load prompt content from general.md
    prompt_file = "general.md"
    if os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as file:
            prompt_content = file.read()
    else:
        raise FileNotFoundError("The general.md file is not found. Please ensure it exists in the current directory.")

    
    ticker = yf.Ticker(stock_code)
    df = ticker.history(period="20d", interval="1h")
    message = f"\n{stock_code} 小时线行情\n{df.to_string()}"
    message += f"\n请分析这个股票的基本面和技术面后,用简洁的语言给出未来一周的操作建议。"

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

    ai_reply = response.choices[0].message.content
    print(ai_reply)

    # 组装完整邮件内容
    import subprocess
    from email.utils import formatdate
    recipient = email_recipient
    sender = "jacobmee@gmail.com"
    subject = f"{stock_code} 行情分析报告"
    date_str = formatdate(localtime=True)
    mail_body = f"{stock_code}【AI分析结论】\n{ai_reply}"
    full_message = f"From: {sender}\nTo: {recipient}\nSubject: {subject}\nDate: {date_str}\nContent-Type: text/plain; charset=utf-8\n\n{mail_body}"

    print("Sending email via msmtp...")
    process = subprocess.Popen([
        '/usr/bin/msmtp', recipient
    ], stdin=subprocess.PIPE)
    process.communicate(input=full_message.encode('utf-8'))
    if process.returncode != 0:
        error_msg = f"/usr/bin/msmtp failed with status code {process.returncode}. Check msmtp debug output above for details."
        print(error_msg)
        raise Exception(error_msg)

    
def main():
    """Run ask_deepseek from command line."""
    ask_deepseek("QQQ","jacob@mitang.me")

if __name__ == "__main__":
    main()
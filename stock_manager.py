from copy import deepcopy
from datetime import datetime
import logging
import logging.handlers
import os
import json
from openai import OpenAI
import pandas as pd

# 调用 deepseek 函数
from Ashare import get_price
from openai import OpenAI

def setup_logger(name):
    """Configure and return a logger with syslog handler."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    logger.handlers = []

    # Create syslog handler
    syslog_handler = logging.handlers.SysLogHandler(address="/dev/log")
    syslog_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(name)s: %(levelname)s - %(message)s")
    syslog_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(syslog_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger

logger = setup_logger("STOCK.MANAGER")



def test_deepseek():
    # Test the Deepseek API integration
    try:
        logger.info("Deepseek API response:")
    except Exception as e:
        logger.error(f"Error testing Deepseek API: {e}")

def ask_deepseek():
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

    message += f"\n你的仓位和交易记录在这里:\n{trade_ops_json_string}"
    message += f"\n现在告诉我你的买卖决策，按照这个json文件的格式给我输出"

    logger.info("Sending prompt to Deepseek:")
    logger.info(prompt_content)
    logger.info("Sending message to Deepseek:")
    logger.info(message)
    
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": prompt_content},
            {"role": "user", "content": message},
        ],
        stream=False
    )
    
    logger.info(response.choices[0].message.content)


    # Save response content to a file named with the current date (YYYY-MMDD.json)
    from datetime import datetime
    today_str = datetime.now().strftime('%Y%m%d')
    filename = f'ai_{today_str}.json'

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            response_content = response.choices[0].message.content
            # Assuming the response content is a valid JSON string
            json_data = json.loads(response_content)
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Response saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving response to {filename}: {e}")

    update_trade_ops_from_ai_file(filename)

    # 发送邮件通知
    try:
        import subprocess
        subject = f"AI.Stocking 日常报告 - {today_str}"
        recipient = "jacob@mitang.me"
        # 尝试将response_content格式化为可读性强的文本
        try:
            json_data = json.loads(response_content)
            lines = []
            # 标题
            lines.append("auto.stocking AI 任务完成报告\n")
            # stock_operations
            if "stock_operations" in json_data:
                lines.append("【操作建议】\n")
                for op in json_data["stock_operations"]:
                    lines.append(f"- 日期: {op.get('date','')}  类型: {op.get('type','')}  股票: {op.get('code','')}  数量: {op.get('amount','')}  价格: {op.get('price','')}  备注: {op.get('comment','')}")
                lines.append("")
            # comment
            if "comment" in json_data:
                lines.append(f"【AI 总结】\n{json_data['comment']}\n")
            # portfolio_analysis
            if "portfolio_analysis" in json_data:
                lines.append("【组合分析】\n")
                for pa in json_data["portfolio_analysis"]:
                    lines.append(f"- 日期: {pa.get('date','')}  总市值: {pa.get('total_value','')}  持仓明细: {pa.get('holdings_detail','')}")
                lines.append("")
            mail_body = '\n'.join(lines)
        except Exception as e:
            mail_body = response_content  # fallback: 原始内容
        # 调用mail命令发送邮件
        process = subprocess.Popen([
            'mail', '-s', subject, recipient
        ], stdin=subprocess.PIPE)
        process.communicate(input=mail_body.encode('utf-8'))
        logger.info(f"Mail sent to {recipient}")
    except Exception as e:
        logger.error(f"Error sending mail: {e}")


def update_trade_ops_from_ai_file(ai_trade_file, trade_ops_file="trade_ops.json"):
    """
    Update trade_ops.json by appending new transactions and a new snapshot if there are real trades in the AI file.
    Only non-HOLD operations are considered real trades. The snapshot format follows trade_ops.example.json.
    """
    try:
        # Load AI trade file
        with open(ai_trade_file, "r", encoding="utf-8") as f:
            ai_data = json.load(f)
        stock_ops = ai_data.get("stock_operations", [])
        real_trades = [op for op in stock_ops if op.get("type") not in ("HOLD", "hold")]
        if not real_trades:
            logger.info(f"No real trades in {ai_trade_file}, trade_ops.json not updated.")
            return False


        # Load current trade_ops.json
        if os.path.exists(trade_ops_file):
            with open(trade_ops_file, "r", encoding="utf-8") as f:
                trade_ops = json.load(f)
        else:
            trade_ops = {"transactions": [], "snapshots": []}

        # Get last snapshot and holdings
        last_snapshot = deepcopy(trade_ops["snapshots"][-1]) if trade_ops["snapshots"] else {
            "timestamp": None, "total_value": 0, "cash": 0, "trade_info": None, "comment": "", "holdings": {}
        }
        holdings = deepcopy(last_snapshot.get("holdings", {}))
        cash = last_snapshot.get("cash", 0)

        # Append new transactions
        for op in real_trades:
            trade_ops["transactions"].append({
                "date": op["date"],
                "type": op["type"],
                "code": op["code"],
                "amount": op["amount"],
                "price": op["price"],
                "comment": op.get("comment", "")
            })

            code = op["code"]
            amount = op["amount"]
            price = op["price"]
            ttype = op["type"].upper()
            # Update holdings and cash
            if ttype == "BUY":
                # Update or create holding
                if code in holdings:
                    prev_shares = holdings[code]["shares"]
                    prev_cost = holdings[code]["cost_basis"]
                    new_shares = prev_shares + amount
                    # Weighted average cost
                    if new_shares > 0:
                        holdings[code]["cost_basis"] = round((prev_shares * prev_cost + amount * price) / new_shares, 2)
                    holdings[code]["shares"] = new_shares
                else:
                    holdings[code] = {"shares": amount, "cost_basis": price, "current_price": price, "market_value": amount * price}
                cash -= amount * price
            elif ttype == "SELL":
                if code in holdings:
                    prev_shares = holdings[code]["shares"]
                    holdings[code]["shares"] = max(prev_shares - amount, 0)
                    # cost_basis remains unchanged for simplicity
                    if holdings[code]["shares"] == 0:
                        holdings[code]["market_value"] = 0
                    else:
                        holdings[code]["market_value"] = holdings[code]["shares"] * price
                cash += amount * price
            # Update current_price and market_value for this code
            if code in holdings:
                holdings[code]["current_price"] = price
                holdings[code]["market_value"] = holdings[code]["shares"] * price

        # Clean up holdings: remove zero-share stocks
        holdings = {k: v for k, v in holdings.items() if v["shares"] > 0}

        # Calculate total_value
        total_value = cash + sum(v["market_value"] for v in holdings.values())

        # trade_info: 所有本次 real_trades 的交易都放入数组
        trade_info = []
        for trade in real_trades:
            trade_info.append({
                "type": trade["type"],
                "code": trade["code"],
                "amount": trade["amount"],
                "price": trade["price"],
                "cost": round(trade["amount"] * trade["price"], 2)
            })

        # Snapshot timestamp: 用最后一笔交易的日期
        snapshot_ts = real_trades[-1]["date"]
        # Comment: 用最后一笔交易的 comment
        comment = real_trades[-1].get("comment", "")

        # Build snapshot
        snapshot = {
            "timestamp": snapshot_ts,
            "total_value": round(total_value, 2),
            "cash": round(cash, 2),
            "trade_info": trade_info,
            "comment": comment,
            "holdings": deepcopy(holdings)
        }
        trade_ops["snapshots"].append(snapshot)

        # Save updated trade_ops.json
        with open(trade_ops_file, "w", encoding="utf-8") as f:
            json.dump(trade_ops, f, ensure_ascii=False, indent=4)
        logger.info(f"trade_ops.json updated with {len(real_trades)} trades and new snapshot from {ai_trade_file}.")

        return True
    except Exception as e:
        logger.error(f"Error updating trade_ops.json from {ai_trade_file}: {e}")
        return False
    
def main():
    """Run ask_deepseek from command line."""
    logger.info("Running deepseek from main...")
    try:
        #test_deepseek()
        ask_deepseek()
        #update_trade_ops_from_ai_file("ai_20251028.json")
    except Exception as e:
        logger.error(f"Error in main calling deepseek: {e}")

if __name__ == "__main__":
    main()
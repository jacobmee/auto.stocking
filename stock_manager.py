from copy import deepcopy
from datetime import datetime
import logging
import os
import json
from openai import OpenAI
import pandas as pd

# 调用 deepseek 函数
from Ashare import get_price
from openai import OpenAI
    
# Setup logger for this module
logger = logging.getLogger("STOCK.MANAGER")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)

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
    filename = f'trade_{today_str}.json'

    update_trade_ops_from_ai_file(filename)

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            response_content = response.choices[0].message.content
            # Assuming the response content is a valid JSON string
            json_data = json.loads(response_content)
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Response saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving response to {filename}: {e}")


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

        # Use the last trade as trade_info for the snapshot
        last_trade = real_trades[-1]
        trade_info = {
            "type": last_trade["type"],
            "code": last_trade["code"],
            "amount": last_trade["amount"],
            "price": last_trade["price"],
            "cost": round(last_trade["amount"] * last_trade["price"], 2)
        }

        # Snapshot timestamp: use last trade date
        snapshot_ts = last_trade["date"]
        # Comment: use last trade comment
        comment = last_trade.get("comment", "")

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
    logger.info("Running ask_deepseek from main...")
    try:
        ask_deepseek()
    except Exception as e:
        logger.error(f"Error in main calling ask_deepseek: {e}")

if __name__ == "__main__":
    main()
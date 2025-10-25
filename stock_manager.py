import json
from datetime import datetime
from typing import List, Dict, Any

class StockManager:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data = {"holdings": {}, "operations": []}
        self._load()

    def _load(self):
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = {"holdings": {}, "operations": []}

    def _save(self):
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def buy(self, code: str, amount: int, comment: str = ""):
        self.data["holdings"][code] = self.data["holdings"].get(code, 0) + amount
        op = {
            "type": "BUY",
            "code": code,
            "amount": amount,
            "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "comment": comment[:250]
        }
        self.data["operations"].append(op)
        self._save()

    def sell(self, code: str, amount: int, comment: str = ""):
        current = self.data["holdings"].get(code, 0)
        if amount > current:
            raise ValueError(f"Not enough shares to sell: {current} available, {amount} requested.")
        self.data["holdings"][code] = current - amount
        op = {
            "type": "SELL",
            "code": code,
            "amount": amount,
            "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "comment": comment[:250]
        }
        self.data["operations"].append(op)
        self._save()

    def get_holdings(self) -> Dict[str, int]:
        return self.data["holdings"]

    def get_operations(self) -> List[Dict[str, Any]]:
        return self.data["operations"]

if __name__ == "__main__":
    manager = StockManager("stock_data.json")
    # 示例操作
    manager.buy("000300", 300, "首次买入沪深300")
    manager.sell("000300", 20, "部分卖出")
    print("当前持仓：", manager.get_holdings())
    print("操作记录：", manager.get_operations())

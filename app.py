from flask import Flask, render_template, jsonify
import os
from line_utils import get_normalized_line, get_value_line
import pandas as pd
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/lines')
def api_lines():
    # 获取000300.XSHG和000688.XSHG小时线
    df1 = get_price_df('000300.XSHG', frequency='60m', count=60)
    df2 = get_price_df('000688.XSHG', frequency='60m', count=60)
    labels1 = [str(x) for x in df1.index] if df1 is not None else []
    labels2 = [str(x) for x in df2.index] if df2 is not None else []
    # 先构造对象数组，便于注入点评
    def build_price_objs(prices):
        return [{"value": v} for v in prices] if prices is not None else []
    def round2(arr):
        return [round(float(x), 2) for x in arr]
    line1_objs = build_price_objs(round2(normalize_prices(df1['close'].values)) if df1 is not None else [])
    line2_objs = build_price_objs(round2(normalize_prices(df2['close'].values)) if df2 is not None else [])
    price1_objs = build_price_objs(df1['close'].values if df1 is not None else [])
    price2_objs = build_price_objs(df2['close'].values if df2 is not None else [])
    # 读取真实交易操作数据
    try:
        with open('trade_ops.json', 'r', encoding='utf-8') as f:
            ops = json.load(f)
    except Exception:
        ops = []
    # 读取今日点评
    try:
        with open('trade_today.json', 'r', encoding='utf-8') as f:
            today_reviews = json.load(f)
    except Exception:
        today_reviews = []
    labels3, line3, value_prices = get_value_line_with_prices(ops, '000300.XSHG', frequency='60m', count=60)
    # 统一labels
    labels = labels1 if labels1 else labels2 if labels2 else labels3
    # 新增：线的名字和代码映射
    line_meta = [
        {"name": "沪深300", "code": "000300.XSHG"},
        {"name": "科创50", "code": "000688.XSHG"},
        {"name": "Deepseek", "code": "Deepseek"}
    ]
    # 新增：将今日点评按股票代码和时间分配到各自曲线节点的comment
    def inject_comments_to_price(price_arr, code, labels):
        # code: 股票代码，labels: x轴时间字符串数组
        if not today_reviews or not price_arr or not labels:
            return
        for review in today_reviews:
            ts = str(review.get('timestamp'))
            idx = None
            for i, t in enumerate(labels):
                if str(t)[:16] == ts[:16]:
                    idx = i
                    break
            if idx is not None:
                for stock in review.get('stock_reviews', []):
                    stock_code = str(stock.get('code'))
                    # 支持完整代码和6位代码匹配
                    if stock_code == code or (stock_code.endswith('.XSHG') and stock_code == code) or (stock_code == code[-6:]) or (code[-6:] == stock_code[-6:]):
                        cmt = stock.get('view')
                        if cmt:
                            if isinstance(price_arr[idx], dict):
                                old = price_arr[idx].get('comment','')
                                price_arr[idx]['comment'] = (old+'\n' if old else '') + cmt
                            else:
                                price_arr[idx] = {
                                    'value': price_arr[idx],
                                    'comment': cmt
                                }
    # 注入点评到各自曲线
    inject_comments_to_price(value_prices, '000300.XSHG', labels)
    inject_comments_to_price(price1_objs, '000300.XSHG', labels)
    inject_comments_to_price(price2_objs, '000688.XSHG', labels)
    # line3归一化也保留两位小数
    line3_rounded = [round(float(x), 2) for x in line3]
    return jsonify({
        "labels": labels,
        "line1": [x["value"] for x in line1_objs],
        "line2": [x["value"] for x in line2_objs],
        "line3": line3_rounded,
        "price1": price1_objs,
        "price2": price2_objs,
        "price3": value_prices,
        "lineMeta": line_meta
    })

# 新增辅助函数
def get_price_df(code, frequency='60m', count=60):
    try:
        from Ashare import get_price
        df = get_price(code, frequency=frequency, count=count)
        return df
    except Exception:
        return None

def normalize_prices(prices):
    if len(prices) == 0:
        return []
    base = prices[0]
    return [1000 * v / base for v in prices]

def get_value_line_with_prices(ops, code, frequency='60m', count=60):
    df = get_price_df(code, frequency=frequency, count=count)
    print('DEBUG 市值主行情长度:', len(df) if df is not None else None)
    if df is not None:
        print('DEBUG 市值主行情时间区间:', df.index[0] if len(df) else None, df.index[-1] if len(df) else None)
    if df is None or len(df) == 0:
        print('DEBUG 市值行情数据为空')
        return [], [], []
    close = df['close'].values
    dates = df.index
    value = []
    value_prices = []
    # 获取所有涉及的股票代码
    all_codes = set(op['code'] for op in ops)
    # 获取所有股票的行情
    code_close = {}
    for c in all_codes:
        dft = get_price_df(c, frequency=frequency, count=count)
        print(f'DEBUG {c} 行情长度:', len(dft) if dft is not None else None)
        if dft is not None and len(dft):
            print(f'DEBUG {c} 行情时间区间:', dft.index[0], dft.index[-1])
        code_close[c] = dft['close'].values if dft is not None and len(dft) == len(dates) else [0]*len(dates)
    # 初始化持仓
    shares = {c: 0 for c in all_codes}
    
    # 现金计算修正点：
    # 将初始现金固定为 100,000，所有交易都在循环中处理现金变动
    INIT_CASH_AMOUNT = 100000
    cash = INIT_CASH_AMOUNT
    
    op_idx = 0
    # 新增：为每个时间点记录comment（如有交易发生）
    last_op_idx = 0
    for i, date in enumerate(dates):
        comments = []
        # 执行当天所有操作，包括第一笔操作，同时更新 cash
        while op_idx < len(ops) and str(ops[op_idx]['date']) <= str(date):
            op = ops[op_idx]
            c = op['code']
            if op['type'] == 'BUY':
                shares[c] += op['amount']
                cash -= op['amount'] * op['price'] 
            elif op['type'] == 'SELL':
                shares[c] -= op['amount']
                cash += op['amount'] * op['price']
            # 新增：如有comment，记录
            if 'comment' in op and op['comment']:
                comments.append(op['comment'])
            op_idx += 1
        # 计算所有股票的市值 (使用当前时间点的收盘价)
        stock_value = {c: shares[c] * code_close[c][i] for c in all_codes}
        total_stock_value = sum(stock_value.values())
        v = cash + total_stock_value
        # 记录结果
        node = {
            "total": round(v,2),
            "cash": round(cash,2),
            "stocks": {c: {"shares": shares[c], "price": round(code_close[c][i],2), "value": round(stock_value[c],2)} for c in all_codes}
        }
        if comments:
            node["comment"] = "\n".join(comments)
        value.append(v)
        value_prices.append(node)
        
    # 修正：将 return 语句移到循环外面 (继承上一个问题的修正)
    base = value[0] if value else 1
    norm = [1000 * v / base for v in value]
    labels = [str(idx) for idx in dates]
    
    return labels, norm, value_prices


if __name__ == '__main__':
    app.run(debug=True, port=5005, host='0.0.0.0')


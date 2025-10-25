import pandas as pd
from Ashare import get_price

def get_normalized_line(code, frequency='60m', count=120):
    df = get_price(code, frequency=frequency, count=count)
    close = df['close'].values
    if len(close) == 0:
        return [], []
    base = close[0]
    norm = [1000 * v / base for v in close]
    labels = [str(idx) for idx in df.index]
    return labels, norm

def get_value_line(ops, code, frequency='60m', count=120):
    # ops: [{date, type, amount, price}]
    df = get_price(code, frequency=frequency, count=count)
    close = df['close'].values
    dates = df.index
    value = []
    shares = 0
    cash = 1000
    op_idx = 0
    for i, date in enumerate(dates):
        # 执行当天所有操作
        while op_idx < len(ops) and str(ops[op_idx]['date']) <= str(date):
            op = ops[op_idx]
            if op['type'] == 'BUY':
                shares += op['amount']
                cash -= op['amount'] * close[i]
            elif op['type'] == 'SELL':
                shares -= op['amount']
                cash += op['amount'] * close[i]
            op_idx += 1
        value.append(cash + shares * close[i])
    base = value[0] if value else 1
    norm = [1000 * v / base for v in value]
    labels = [str(idx) for idx in dates]
    return labels, norm

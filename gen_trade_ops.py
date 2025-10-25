import json
from Ashare import get_price
import numpy as np
from datetime import datetime, timedelta

# 获取真实行情数据
code1 = '000300.XSHG'
code2 = '000688.XSHG'
count = 60

df1 = get_price(code1, frequency='60m', count=count)
df2 = get_price(code2, frequency='60m', count=count)



# 生成真实风格的交易，初始金额100000，首次买入金额大于70000，然后再进行8笔分散交易
ops = []
init_cash = 100000
cash = init_cash
shares = {code1: 0, code2: 0}
np.random.seed(42)

# 首次买入，金额>70000，分配给code1和code2
row1 = df1.iloc[0]
row2 = df2.iloc[0]
price1 = float(np.random.uniform(row1['low'], row1['high']))
price2 = float(np.random.uniform(row2['low'], row2['high']))

# 随机分配大额买入比例
portion1 = np.random.uniform(0.4, 0.6)
portion2 = 1 - portion1
buy_amt1 = int((init_cash * portion1) // price1)
buy_amt2 = int((init_cash * portion2) // price2)
amt1 = max(1, buy_amt1)
amt2 = max(1, buy_amt2)
amt1_val = amt1 * price1
amt2_val = amt2 * price2
while amt1_val + amt2_val < 70000:
    amt1 += 1
    amt1_val = amt1 * price1
    if amt1_val + amt2_val > init_cash:
        amt1 -= 1
        break
while amt1_val + amt2_val < 70000:
    amt2 += 1
    amt2_val = amt2 * price2
    if amt1_val + amt2_val > init_cash:
        amt2 -= 1
        break

amt1 = int(amt1)
amt2 = int(amt2)
amt1_val = amt1 * price1
amt2_val = amt2 * price2

if amt1 > 0:
    cash -= amt1_val
    shares[code1] += amt1
    ops.append({
        'date': df1.index[0].strftime('%Y-%m-%d %H:%M:%S'),
        'type': 'BUY',
        'code': code1,
        'amount': amt1,
        'price': round(price1,2),
        'comment': f'首次大额买入{code1}'
    })
if amt2 > 0:
    cash -= amt2_val
    shares[code2] += amt2
    ops.append({
        'date': df2.index[0].strftime('%Y-%m-%d %H:%M:%S'),
        'type': 'BUY',
        'code': code2,
        'amount': amt2,
        'price': round(price2,2),
        'comment': f'首次大额买入{code2}'
    })

# 选8个分布在30天内的点，均匀分布，跳过第一个点
idxs = np.linspace(1, count-1, 8, dtype=int)
dates = [df1.index[i] for i in idxs]

for i, dt in enumerate(dates):
    if i % 2 == 0:
        code = code1
        df = df1
    else:
        code = code2
        df = df2
    row = df.iloc[idxs[i]]
    price = float(np.random.uniform(row['low'], row['high']))
    # 买入或卖出
    if i < 4:
        # 买入，保证买入后cash剩余不低于30%
        max_amount = int((cash - 0.3 * init_cash) // price)
        amount = np.random.randint(1, max(2, max_amount+1)) if max_amount > 0 else 0
        if amount > 0 and (cash - amount * price) >= 0.3 * init_cash:
            cash -= amount * price
            shares[code] += amount
            ops.append({
                'date': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'BUY',
                'code': code,
                'amount': amount,
                'price': round(price,2),
                'comment': f'低位买入{code}'
            })
    else:
        # 卖出时不能超过持仓
        max_amount = shares[code]
        amount = np.random.randint(1, max(2, max_amount+1)) if max_amount > 0 else 0
        if amount > 0:
            cash += amount * price
            shares[code] -= amount
            ops.append({
                'date': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'SELL',
                'code': code,
                'amount': amount,
                'price': round(price,2),
                'comment': f'高位卖出{code}'
            })

with open('trade_ops.json', 'w', encoding='utf-8') as f:
    json.dump(ops, f, ensure_ascii=False, indent=2)
print('已生成真实行情基础的模拟交易数据，见 trade_ops.json')

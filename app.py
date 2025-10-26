from flask import send_from_directory
from flask import Flask, render_template, jsonify, request
import os
import sys # 修正：导入 sys 模块以解决 Ashare 数据获取时的依赖问题
# from line_utils import get_normalized_line, get_value_line # 假设这些已在外部或被替换
import pandas as pd
import json

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

# 提供 custom_code.json 静态访问接口
@app.route('/custom_code.json')
def custom_code_json():
    return send_from_directory('.', 'custom_code.json')


@app.route('/api/save_custom_codes', methods=['POST'])
def save_custom_codes():
    try:
        codes = request.get_json()
        
        if not isinstance(codes, list):
            return jsonify({"error": "Data must be a JSON array."}), 400

        file_path = 'custom_code.json'
        
        # 核心逻辑：直接用新数据覆盖文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(codes, f, indent=4)
        
        return jsonify({"message": "Custom codes successfully saved."}), 200

    except Exception as e:
        print(f"Error saving custom codes: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/api/lines')
def api_lines():
    # 获取基准日期参数
    base_date = request.args.get('base_date')
    # 获取自定义股票参数
    import json as _json
    codes_str = request.args.get('codes', '')
    input_codes = [c.strip() for c in codes_str.split(',') if c.strip()] if codes_str else []
    # 自动补全后缀并轮询
    suffixes = ['.XSHG', '.XSHE', '.XHSG', '.HK']
    checked_codes = []
    code_map = {}  # 原始输入 -> 实际有行情的代码
    for code in input_codes:
        if '.' in code:
            checked_codes.append(code)
            print(f"自定义股票尝试: {code} (已带后缀，直接查)")
        elif code.isdigit():
            found = False
            for suf in suffixes:
                try_code = code + suf
                df = get_price_df(try_code, frequency='60m', count=60)
                print(f"自定义股票尝试: {try_code} -> {'有数据' if (df is not None and len(df)) else '无数据'}")
                if df is not None and len(df):
                    checked_codes.append(try_code)
                    code_map[code] = try_code
                    found = True
                    break
            if not found:
                code_map[code] = None
        else:
            code_map[code] = None
    # 只有 codes 参数非空时才写 custom_code.json，避免页面刷新时误清空
    if input_codes:
        try:
            with open('custom_code.json', 'w', encoding='utf-8') as f:
                _json.dump(input_codes, f, ensure_ascii=False)
        except Exception as e:
            print('写custom_code.json失败:', e)
    
    # 查找基准点索引
    def get_base_idx(df, base_date):
        """根据 base_date 查找 DataFrame 中的时间索引，如果找不到返回 -1。"""
        # 如果没有基准日期或数据为空，默认返回 0（但我们更希望知道是否找到了）
        if not base_date or df is None or len(df) == 0:
            return 0
        # 查找与 base_date 匹配的第一个索引 (按时间点查找)
        for i, t in enumerate(df.index):
            tstr = str(t)
            if tstr[:10] == base_date:
                return i
        # 如果找不到匹配的日期
        return -1 # 返回 -1 明确表示未找到

    # 获取行情
    df1 = get_price_df('000300.XSHG', frequency='60m', count=60)
    df2 = get_price_df('000688.XSHG', frequency='60m', count=60)
    custom_dfs = [get_price_df(code, frequency='60m', count=60) for code in checked_codes]
    
    # 沪深300和科创50的归一化基准索引 (基于时间点)
    base_idx1 = get_base_idx(df1, base_date)
    # 如果 000300 的基准日期找不到，回退到 0
    base_idx1 = base_idx1 if base_idx1 != -1 else 0
    base_idx2 = get_base_idx(df2, base_date)
    base_idx2 = base_idx2 if base_idx2 != -1 else 0
    
    labels1 = [str(x) for x in df1.index] if df1 is not None else []
    labels2 = [str(x) for x in df2.index] if df2 is not None else []
    custom_labels = [[str(x) for x in (df.index if df is not None else [])] for df in custom_dfs]
    
    # 构造对象数组
    def build_price_objs(prices):
        # 允许 prices 中包含 None 来表示缺失点
        if prices is None:
            return []
        # 返回 [{"value": float or None}, ...] 结构
        return [{"value": (round(float(v), 2) if v is not None else None)} for v in prices]
        
    def round2(arr):
        return [round(float(x), 2) for x in arr]

    # 归一化和格式化主线数据
    line1_objs = build_price_objs(round2(normalize_prices(df1['close'].values, base_idx1)) if df1 is not None else [])
    line2_objs = build_price_objs(round2(normalize_prices(df2['close'].values, base_idx2)) if df2 is not None else [])
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
        
    # 市值线基准点索引与labels
    base_idx3 = base_idx1  # 市值线与沪深300对齐
    labels3, line3, value_prices = get_value_line_with_prices(ops, '000300.XSHG', frequency='60m', count=60, base_idx=base_idx1)
    
    # 统一labels（优先沪深300，否则科创50，否则市值，否则第一个自定义）
    labels = labels1 or labels2 or labels3 or (custom_labels[0] if custom_labels else [])
    
    # 市值线归一化
    if line3 and 0 <= base_idx3 < len(line3):
        base3 = line3[base_idx3]
        if base3 == 0:
            base3 = 1
        line3_rounded = [round(float(x) * 1000 / base3, 2) for x in line3]
    else:
        line3_rounded = [round(float(x), 2) for x in line3]
        
    # 点评注入
    def inject_comments_to_price(price_arr, code, labels):
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
                # 检查索引是否在 price_arr 范围内
                if idx < len(price_arr):
                    for stock in review.get('stock_reviews', []):
                        stock_code = str(stock.get('code'))
                        # 兼容处理
                        if stock_code == code or (stock_code.endswith('.XSHG') and stock_code == code) or (stock_code == code[-6:]) or (code[-6:] == stock_code[-6:]):
                            cmt = stock.get('view')
                            if cmt:
                                # price_arr[idx] 保证是 [{"value":...}, {"value":...}] 结构
                                old = price_arr[idx].get('comment','')
                                price_arr[idx]['comment'] = (old+'\n' if old else '') + cmt

    inject_comments_to_price(value_prices, '000300.XSHG', labels)
    inject_comments_to_price(price1_objs, '000300.XSHG', labels)
    inject_comments_to_price(price2_objs, '000688.XSHG', labels)
    
    # 新增辅助函数：安全填充数组
    def safe_pad(arr, target_len, pad_value=None):
        return list(arr) + [pad_value] * (target_len - len(arr))
    
    # 函数：根据 labels[0] 的时间点查找自定义 df 的切片起点索引
    def get_slice_start_idx(df, labels):
        if not labels or df is None or len(df) == 0:
            return 0
        start_time = labels[0]
        for i, t in enumerate(df.index):
            if str(t) == start_time:
                return i
        return 0 # 如果找不到对齐点，从 0 开始切

    # 处理自定义股票线
    custom_lines = []
    custom_prices = []
    custom_meta = []
    for idx, (code, df) in enumerate(zip(checked_codes, custom_dfs)):
        if df is not None and len(df) and len(labels) > 0:
            
            # --- 1. 确定归一化的基准点索引 (Normalization Base Index) ---
            # 尝试使用 base_date 查找基准点
            norm_base_idx = get_base_idx(df, base_date) 
            
            # --- 2. 确定对齐/切片起点索引 (Slice Start Index) ---
            # 确保自定义数据从全局 labels 的起始时间开始切片
            slice_start_idx = get_slice_start_idx(df, labels)
            
            # --- 3. 确定最终用于归一化的价格索引 ---
            price_for_1000_base_idx = norm_base_idx
            
            # 核心修正逻辑：如果 base_date 提供了，但该股票数据中找不到 (-1)，
            # 则回退到使用当前视图的第一个可见点作为归一化基准 (slice_start_idx)。
            if base_date and norm_base_idx == -1:
                price_for_1000_base_idx = slice_start_idx
                print(f"Warning: Base date {base_date} not found for {code}. Falling back to view start for normalization.")
            elif norm_base_idx == -1:
                # 如果没有 base_date，且 get_base_idx 返回 -1 (这不应该发生，因为默认返回 0)，则使用 0
                 price_for_1000_base_idx = 0

            # --- 4. 计算完整归一化序列 ---
            # 确保 prices 数组长度和 base index 有效
            if price_for_1000_base_idx >= len(df['close'].values):
                price_for_1000_base_idx = 0 # 兜底，避免越界

            # 使用 price_for_1000_base_idx 作为归一化基准
            norm_full = normalize_prices(df['close'].values, price_for_1000_base_idx)
            price_full = df['close'].values
            
            # --- 5. 切片并填充数据以匹配全局 labels 长度 ---
            end_idx = slice_start_idx + len(labels)
            
            # 切片
            norm_sliced = norm_full[slice_start_idx:end_idx]
            price_sliced = price_full[slice_start_idx:end_idx]
            
            # 填充 (如果自定义数据比全局标签短，用 None 填充)
            norm_aligned = safe_pad(norm_sliced, len(labels))
            price_aligned = safe_pad(price_sliced, len(labels))

            # --- 6. 格式化和注入点评 ---
            # 安全四舍五入
            norm_aligned_rounded = [round(float(x), 2) if x is not None else None for x in norm_aligned]
            
            price_objs = build_price_objs(price_aligned) # 使用修正后的 build_price_objs
            
            # 注入点评
            inject_comments_to_price(price_objs, code, labels)
            
            custom_lines.append(norm_aligned_rounded) # 使用包含 None 的 rounded 列表
            custom_prices.append(price_objs)
            custom_meta.append({"name": code, "code": code})
        else:
            # 无数据时补空，长度必须与 labels 一致
            custom_lines.append([None] * len(labels)) 
            custom_prices.append([{"value": None}] * len(labels))
            custom_meta.append({"name": code, "code": code})

    line_meta = [
        {"name": "沪深300", "code": "000300.XSHG"},
        {"name": "科创50", "code": "000688.XSHG"},
        {"name": "Deepseek", "code": "Deepseek"}
    ] + custom_meta
    return jsonify({
        "labels": labels,
        "line1": [x["value"] for x in line1_objs],
        "line2": [x["value"] for x in line2_objs],
        "line3": line3_rounded,
        "price1": price1_objs,
        "price2": price2_objs,
        "price3": value_prices,
        "customLines": custom_lines,
        "customPrices": custom_prices,
        "lineMeta": line_meta
    })

def get_price_df(code, frequency='60m', count=60):
    try:
        from Ashare import get_price
        print(f"Fetching data for {code} with frequency {frequency} and count {count}.")
        df = get_price(code, frequency=frequency, count=count)
        if df is None or df.empty:
            print(f"No data returned for {code}.")
        else:
            print(f"Data fetched for {code}: {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error fetching data for {code}: {e}")
        return None

def normalize_prices(prices, base_idx=0):
    if prices is None or len(prices) == 0:
        return []
    if not (0 <= base_idx < len(prices)):
        base_idx = 0
    base = prices[base_idx]
    if base == 0:
        base = 1
    return [1000 * v / base for v in prices]

def get_value_line_with_prices(ops, code, frequency='60m', count=60, base_idx=0):
    df = get_price_df(code, frequency=frequency, count=60)
    print('DEBUG 市值主行情长度:', len(df) if df is not None else None)
    if df is not None:
        print('DEBUG 市值主行情时间区间:', df.index[0] if len(df) else None, df.index[-1] if len(df) else None)
    if df is None or len(df) == 0:
        print('DEBUG 市值行情数据为空')
        return [], [], []
    close = df['close'].values if df is not None else []
    dates = df.index
    value = []
    value_prices = []
    all_codes = set(op['code'] for op in ops)
    code_close = {}
    for c in all_codes:
        dft = get_price_df(c, frequency=frequency, count=60)
        print(f'DEBUG {c} 行情长度:', len(dft) if dft is not None else None)
        if dft is not None and len(dft):
            print(f'DEBUG {c} 行情时间区间:', dft.index[0], dft.index[-1])
        # 确保 code_close[c] 的长度与 dates 长度一致，否则填充 0
        code_close[c] = dft['close'].values if dft is not None and len(dft) == len(dates) else [0]*len(dates)
    shares = {c: 0 for c in all_codes}
    INIT_CASH_AMOUNT = 100000
    cash = INIT_CASH_AMOUNT
    op_idx = 0
    # last_op_idx = 0 # 移除未使用的变量
    for i, date in enumerate(dates):
        comments = []
        while op_idx < len(ops) and str(ops[op_idx]['date']) <= str(date):
            op = ops[op_idx]
            c = op['code']
            # 确保操作的股票代码在 code_close 中有数据
            if c in code_close and i < len(code_close[c]):
                op_price = op['price'] 
                
                # 检查 op['price'] 是否合理，如果不合理使用当时的 close price
                # 这里的逻辑是使用 op['price'] 进行交易，但如果 op['price'] 为 0 或缺失，
                # 应该使用当前的 code_close[c][i] 来估算（但通常 op['price'] 是交易价格，不应随意修改）
                # 保持原逻辑，假设 op['price'] 是交易发生时的价格
                
                if op['type'] == 'BUY':
                    shares[c] += op['amount']
                    cash -= op['amount'] * op_price 
                elif op['type'] == 'SELL':
                    shares[c] -= op['amount']
                    cash += op['amount'] * op_price
                    
            if 'comment' in op and op['comment']:
                comments.append(op['comment'])
            op_idx += 1
            
        stock_value = {c: shares[c] * code_close[c][i] for c in all_codes}
        total_stock_value = sum(stock_value.values())
        v = cash + total_stock_value
        node = {
            "total": round(v,2),
            "cash": round(cash,2),
            "stocks": {c: {"shares": shares[c], "price": round(code_close[c][i],2), "value": round(stock_value[c],2)} for c in all_codes}
        }
        if comments:
            node["comment"] = "\n".join(comments)
        value.append(v)
        value_prices.append(node)
    
    base = value[base_idx] if value and 0 <= base_idx < len(value) else (value[0] if value else 1)
    if base == 0:
        base = 1
        
    norm = [1000 * v / base for v in value]
    labels = [str(idx) for idx in dates]
    return labels, norm, value_prices


if __name__ == '__main__':
    app.run(debug=True, port=5005, host='0.0.0.0')

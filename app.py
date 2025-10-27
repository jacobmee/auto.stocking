from flask import send_from_directory
from flask import Flask, render_template, jsonify, request
import os
import sys # 修正：导入 sys 模块以解决 Ashare 数据获取时的依赖问题
# from line_utils import get_normalized_line, get_value_line # 假设这些已在外部或被替换
import pandas as pd
import json
import threading
import time
import logging
import logging.handlers
from datetime import datetime, timedelta

app = Flask(__name__)

def setup_logger(name):
    """Configure and return a logger with syslog handler."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    logger.handlers = []

    # Create syslog handler
    syslog_handler = logging.handlers.SysLogHandler(address='/dev/log')
    syslog_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(name)s: %(levelname)s - %(message)s')
    syslog_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(syslog_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger

# Get logger
logger = setup_logger('STOCKING')

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
        logger.error(f"Error saving custom codes: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/api/lines')
def api_lines():
    # Remove the default base_date value
    base_date = request.args.get('base_date')
    
    # 获取自定义股票参数
    import json as _json
    codes_str = request.args.get('codes', '')
    input_codes = [c.strip() for c in codes_str.split(',') if c.strip()] if codes_str else []
    # 自动补全后缀并轮询
    suffixes = ['.XSHG', '.XSHE', '.XHSG']
    checked_codes = []
    code_map = {}  # 原始输入 -> 实际有行情的代码
    for code in input_codes:
        if '.' in code:
            checked_codes.append(code)
            logger.info(f"自定义股票尝试: {code} (已带后缀，直接查)")
        elif code.isdigit():
            found = False
            for suf in suffixes:
                try_code = code + suf
                df = get_price_df(try_code, frequency='60m', count=60)
                logger.info(f"自定义股票尝试: {try_code} -> {'有数据' if (df is not None and len(df)) else '无数据'}")
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
            logger.error(f'写custom_code.json失败: {e}')
    
    # 重新从 custom_code.json 读取 checked_codes，确保后续流程用的是最新文件内容
    try:
        with open('custom_code.json', 'r', encoding='utf-8') as f:
            checked_codes = json.load(f)
        logger.info(f"custom_code.json 重新读取自定义股票列表: {checked_codes}")
    except Exception as e:
        logger.error(f"重新读取 custom_code.json 失败: {e}")
        checked_codes = []
    
    logger.info(f"最终使用的自定义股票列表: {checked_codes}")
    
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
    custom_dfs = []
    for code in checked_codes:
        logger.info(f"Fetching custom stock data for {code} (60m, 60 bars)...")
        df = get_price_df(code, frequency='60m', count=60)
        if df is not None and len(df):
            logger.info(f"自定义股票: {code} fetched: {len(df)} rows.")
        else:
            logger.warning(f"自定义股票: {code} 无数据.")
        custom_dfs.append(df)
    
    # 沪深300和科创50的归一化基准索引 (基于时间点)
    base_idx1 = get_base_idx(df1, base_date)
    # 如果 000300 的基准日期找不到，回退到 0
    base_idx1 = base_idx1 if base_idx1 != -1 else 0
    base_idx2 = get_base_idx(df2, base_date)
    base_idx2 = base_idx2 if base_idx2 != -1 else 0
    
    labels1 = [str(x) for x in df1.index] if df1 is not None else []
    labels2 = [str(x) for x in df2.index] if df2 is not None else []
    custom_labels = []
    for idx, df in enumerate(custom_dfs):
        if df is not None:
            labels = [str(x) for x in df.index]
            logger.info(f"自定义股票 Labels: {checked_codes[idx]} labels count: {len(labels)}")
        else:
            labels = []
            logger.warning(f"自定义股票 Labels: {checked_codes[idx]} 无数据，labels 为空。")
        custom_labels.append(labels)
    
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
            if not isinstance(ops, list) or not all(isinstance(op, dict) for op in ops):
                raise ValueError("The content of trade_ops.json must be a list of dictionaries.")
    except Exception as e:
        logger.error(f"Error loading or validating trade_ops.json: {e}")
        ops = []  # Default to an empty list if validation fails
        
    # 读取今日点评

    try:
        from glob import glob
        today_str = datetime.now().strftime('%Y%m%d')
        filename = f'trade_{today_str}.json'
        if os.path.exists(filename):
            trade_file = filename
        else:
            # Search for the most recent trade_YYYY-MMDD.json in the last 10 days
            found = False
            for delta in range(1, 11):
                prev_date = (datetime.now() - timedelta(days=delta)).strftime('%Y-%m%d')
                prev_file = f'trade_{prev_date}.json'
                if os.path.exists(prev_file):
                    trade_file = prev_file
                    found = True
                    break

            if not found:
                logger.warning('No recent trade_YYYYMMDD.json file found in the last 10 days.')
                today_reviews = []
                trade_file = None

        if trade_file:
            logger.info(f'Using trade file: {trade_file}')

            with open(trade_file, 'r', encoding='utf-8') as f:
                today_data = json.load(f)
                today_reviews = today_data.get('stock_view', [])
                if not isinstance(today_reviews, list) or not all(isinstance(review, dict) for review in today_reviews):
                    raise ValueError("The 'stock_view' in trade_YYYY-MMDD.json must be a list of dictionaries.")

                # Ensure each review in today_reviews has a valid timestamp
                today_date = pd.Timestamp.now().strftime('%Y-%m-%d')
                for review in today_reviews:
                    if 'timestamp' not in review or not review['timestamp']:
                        review['timestamp'] = today_date  # Default to today's date if missing
        else:
            today_reviews = []
    except Exception as e:
        logger.error(f"Error loading or validating today_reviews: {e}")
        today_reviews = []  # Default to an empty list if validation fails
        
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
        
    def get_base_code(code_str):
        """提取股票代码的基础6位数字部分。"""
        if not code_str:
            return ""
        # 查找所有数字部分并返回后6位
        import re
        digits = re.findall(r'\d+', code_str)
        if digits:
            return digits[-1][-6:]
        return ""   

    def inject_comments_to_price(price_arr, target_code, labels, today_reviews):
        """
        将点评数据注入到价格数组的 comment 字段中。
        :param price_arr: 图表数据中的价格数组 (e.g., price1_objs)
        :param target_code: 要匹配的股票代码 (e.g., '000300.XSHG')
        :param labels: 图表的X轴时间标签数组
        :param today_reviews: 包含点评信息的列表 (stock_view)
        """
        if not today_reviews or not price_arr or not labels:
            return

        # 1. 预处理目标代码，方便匹配
        target_base_code = get_base_code(target_code)
        
        # 2. 遍历所有点评
        for review in today_reviews:
            if not isinstance(review, dict) or 'view' not in review or 'code' not in review:
                continue
                
            review_code = str(review.get('code'))
            review_ts = str(review.get('date', review.get('timestamp', ''))) # 使用 'date' 或 'timestamp'
            review_base_code = get_base_code(review_code)

            # **代码匹配逻辑优化**：只要基础 6 位代码一致，就认为匹配
            if not target_base_code or target_base_code != review_base_code:
                continue
            
            # 3. 查找时间戳匹配的索引
            idx = None
            if review_ts:
                # 尝试精确匹配 (精确到秒)
                try:
                    idx = labels.index(review_ts)
                except ValueError:
                    # 尝试匹配到分钟 (如果 labels 粒度较粗)
                    ts_minute = review_ts[:16] # 'YYYY-MM-DD HH:MM'
                    for i, t in enumerate(labels):
                        if str(t)[:16] == ts_minute:
                            idx = i
                            break
            else:
                # 如果时间戳缺失，默认注入到最后一个点
                idx = len(price_arr) - 1

            # 4. 注入点评
            if idx is not None and 0 <= idx < len(price_arr):
                cmt = review.get('view')
                if cmt:
                    # 确保 price_arr[idx] 是一个 dict，并且有 comment 字段
                    if not isinstance(price_arr[idx], dict):
                        # 如果不是 dict，尝试修复或跳过 (这里假设后端逻辑保证它是 dict)
                        price_arr[idx] = {'value': price_arr[idx]} # 假设它是裸值
                    
                    old = price_arr[idx].get('comment', '')
                    
                    # **核心修复**：使用 'date' 字段的时间，保持一致性
                    date_prefix = review_ts if review_ts else "" 
                    
                    # 注入点评内容
                    price_arr[idx]['comment'] = (old + '\n' if old else '') + cmt
                    
                    # 可以在这里添加一个标记，确认注入成功
                    # print(f"Injected comment for {target_code} at index {idx}")


    # 示例调用 (假设 today_reviews, value_prices, price1_objs, price2_objs, labels 都已定义)
    # 假设 today_reviews = result_data.get('stock_view', [])

    # 假设以下调用是正确的
    inject_comments_to_price(price1_objs, '000300.XSHG', labels, today_reviews)
    inject_comments_to_price(price2_objs, '000688.XSHG', labels, today_reviews)

    
    # Reorganize /api/lines to include Portfolio Analysis, Stock Operations, and Stock Views in tooltips
    for i, node in enumerate(value_prices):
        # Embed Portfolio Analysis
        if i == len(value_prices) - 1:  # Add Portfolio Analysis to the last time point
            portfolio_comment = today_data.get('portfolio_analysis', {}).get('reasoning', '')
            if portfolio_comment:
                node['portfolio_analysis'] = {
                    "market_summary": today_data['portfolio_analysis'].get('market_summary', ''),
                    "risk_level": today_data['portfolio_analysis'].get('risk_level', ''),
                    "overall_operation": today_data['portfolio_analysis'].get('overall_operation', ''),
                    "trading_plan": today_data['portfolio_analysis'].get('trading_plan', ''),
                    "portfolio_comment": portfolio_comment,
                    "date": today_data['portfolio_analysis'].get('date', '')
                }

        # Embed Stock Operations
        for operation in today_data.get('stock_operations', []):
            if str(operation['date'])[:16] == str(labels[i])[:16]:
                if 'stock_operations' not in node:
                    node['stock_operations'] = []
                node['stock_operations'].append({
                    "date": operation['date'],
                    "type": operation['type'],
                    "code": operation['code'],
                    "amount": operation['amount'],
                    "price": operation['price'],
                    "comment": operation['comment']
                })


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
                logger.warning(f"Base date {base_date} not found for {code}. Falling back to view start for normalization.")
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
            inject_comments_to_price(price_objs, code, labels, "")
            
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
        logger.info(f"Fetching data for {code} with frequency {frequency} and count {count}.")
        df = get_price(code, frequency=frequency, count=count)
        if df is None or df.empty:
            logger.warning(f"No data returned for {code}.")
        else:
            logger.info(f"Data fetched for {code}: {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {code}: {e}")
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
    logger.debug(f'DEBUG 市值主行情长度: {len(df) if df is not None else None}')
    if df is not None:
        logger.debug(f'DEBUG 市值主行情时间区间: {df.index[0] if len(df) else None}, {df.index[-1] if len(df) else None}')
    if df is None or len(df) == 0:
        logger.debug('DEBUG 市值行情数据为空')
        return [], [], []
    close = df['close'].values if df is not None else []
    dates = df.index
    value = []
    value_prices = []
    all_codes = set(op['code'] for op in ops)
    code_close = {}
    for c in all_codes:
        dft = get_price_df(c, frequency=frequency, count=60)
        logger.debug(f'DEBUG {c} 行情长度: {len(dft) if dft is not None else None}')
        if dft is not None and len(dft):
            logger.debug(f'DEBUG {c} 行情时间区间: {dft.index[0]}, {dft.index[-1]}')
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
        # Ensure comments are added to the value_prices array
        if comments:
            existing_comment = node.get("comment", "")
            node["comment"] = (existing_comment + "\n" if existing_comment else "") + "\n".join(comments)
        value.append(v)
        value_prices.append(node)
    
    base = value[base_idx] if value and 0 <= base_idx < len(value) else (value[0] if value else 1)
    if base == 0:
        base = 1
        
    norm = [1000 * v / base for v in value]
    labels = [str(idx) for idx in dates]
    return labels, norm, value_prices


# Process trade_ops.json snapshots
try:
    with open('trade_ops.json', 'r', encoding='utf-8') as f:
        trade_ops = json.load(f)
        snapshots = trade_ops.get('snapshots', [])
        for snapshot in snapshots:
            timestamp = snapshot.get('timestamp', 'N/A')
            comment = snapshot.get('comment', '')
            holdings = snapshot.get('holdings', {})
            logger.info(f"Snapshot at {timestamp}: {comment}")
            logger.info(f"Holdings: {holdings}")
except Exception as e:
    logger.error(f"Error processing trade_ops.json: {e}")

# Process trade_today.json portfolio analysis and stock operations
try:
    with open('trade_today.json', 'r', encoding='utf-8') as f:
        trade_today = json.load(f)
        portfolio_analysis = trade_today.get('portfolio_analysis', {})
        stock_operations = trade_today.get('stock_operations', [])
        stock_view = trade_today.get('stock_view', [])

        # Print portfolio analysis summary
        logger.info("Portfolio Analysis:")
        logger.info(f"Market Summary: {portfolio_analysis.get('market_summary', '')}")
        logger.info(f"Risk Level: {portfolio_analysis.get('risk_level', '')}")
        logger.info(f"Overall Operation: {portfolio_analysis.get('overall_operation', '')}")
        logger.info(f"Trading Plan: {portfolio_analysis.get('trading_plan', '')}")

        # Print stock operations
        logger.info("Stock Operations:")
        for operation in stock_operations:
            logger.info(operation)

        # Print stock views
        logger.info("Stock Views:")
        for view in stock_view:
            logger.info(view)
except Exception as e:
    logger.error(f"Error processing trade_today.json: {e}")


def run_daily_task_at(hour, minute, task_func):
    def scheduler():
        while True:
            now = datetime.now()
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now >= next_run:
                next_run += timedelta(days=1)
            sleep_seconds = (next_run - now).total_seconds()
            time.sleep(sleep_seconds)
            try:
                task_func()
            except Exception as e:
                logger.error(f"Scheduled task error: {e}")
    t = threading.Thread(target=scheduler, daemon=True)
    t.start()

# 示例：定义你要定时执行的任务
import importlib

def my_daily_task():
    try:
        stock_manager = importlib.import_module('stock_manager')
        if hasattr(stock_manager, 'test_deepseek'):
            stock_manager.test_deepseek()
        else:
            logger.warning('test_deepseek not found in stock_manager.py')
    except Exception as e:
        logger.error(f"Error running test_deepseek: {e}")


# 每天下午 14:15 执行 ask_deepseek
def my_daily_task():
    try:
        stock_manager = importlib.import_module('stock_manager')
        if hasattr(stock_manager, 'ask_deepseek'):
            logger.warning('RUNNING: ask_deepseek daily task...')
            stock_manager.ask_deepseek()
        else:
            logger.warning('ask_deepseek not found in stock_manager.py')
    except Exception as e:
        logger.error(f"Error running ask_deepseek: {e}")

run_daily_task_at(14, 15, my_daily_task)

if __name__ == '__main__':
    app.run(debug=True, port=5005, host='0.0.0.0')

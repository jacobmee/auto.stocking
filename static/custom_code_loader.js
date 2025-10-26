// 读取 custom_code.json 并自动填充自定义股票输入框
async function loadCustomCodesAndFill() {
    try {
        const resp = await fetch('/custom_code.json?_t=' + Date.now()); // 防缓存
        if (!resp.ok) {
            console.log('custom_code.json fetch failed:', resp.status);
            return;
        }
        const codes = await resp.json();
        console.log('custom_code.json 读取结果:', codes);
        if (Array.isArray(codes)) {
            if (typeof customStocks !== 'undefined') {
                customStocks = codes.slice(0, 5);
                renderCustomStockList();
                triggerChartUpdate && triggerChartUpdate();
            }
        }
    } catch (e) {
        console.log('custom_code.json 读取异常:', e);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    loadCustomCodesAndFill();
});

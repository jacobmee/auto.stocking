/**
 * custom_code_save.js
 * * 负责将当前的 customStocks 数组保存到服务器的逻辑。
 * 依赖于主脚本中定义的全局变量 customStocks。
 */

// 假设主脚本中有一个全局数组 `customStocks` 存储了自定义股票代码

/**
 * 将当前的自定义股票代码列表保存到服务器。
 */
window.saveCustomCodes = async function() {
    // 检查 customStocks 是否存在且为数组
    if (typeof customStocks === 'undefined' || !Array.isArray(customStocks)) {
        console.error('保存失败: 全局变量 customStocks 未定义或不是数组。');
        return;
    }

    // 假设保存的 API 路由是 /api/save_custom_codes
    const apiUrl = '/api/save_custom_codes';

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                // 如果需要认证，可以在这里添加认证头，例如 'Authorization': 'Bearer YOUR_TOKEN'
            },
            // 将股票代码数组转换为 JSON 字符串作为请求体
            body: JSON.stringify(customStocks)
        });

        if (response.ok) {
            console.log('自定义股票代码保存成功。', customStocks);
            // 可以选择在这里显示一个成功的提示
            // alert('保存成功!'); 
        } else {
            // 尝试读取服务器返回的错误信息
            const errorText = await response.text();
            console.error(`自定义股票代码保存失败。状态码: ${response.status}. 详情: ${errorText}`);
            // alert(`保存失败 (${response.status}): ${errorText.substring(0, 50)}...`);
        }
    } catch (error) {
        console.error('保存自定义股票代码时发生网络或请求错误:', error);
        // alert('保存请求失败，请检查网络连接。');
    }
};

// 可以在这里添加一个简单的测试调用（可选，用于调试）
// document.addEventListener('DOMContentLoaded', () => {
//     // 可以在这里测试 saveCustomCodes，但通常只在用户操作时调用
// });
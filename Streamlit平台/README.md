# 农行杯 Streamlit 平台

## 一、快速启动

```bash
# 1. 进入文件夹
cd Streamlit平台

# 2. 安装依赖（只需一次）
pip install -r requirements.txt

# 3. 启动
streamlit run app.py

# 4. 浏览器打开显示的地址（通常是 http://localhost:8501）
```

## 二、页面说明

| 页面 | 内容 | 亮点 |
|------|------|------|
| 📈 市场仪表盘 | 四维评分、雷达图、历史走势、状态分布 | 日期滑块可查看任意历史日期的评分 |
| 📊 策略对比 | 8 套策略指标表、净值曲线、分段回测 | 多选框筛选策略，高亮最佳行 |
| 🎯 配置推荐 | 三档风险组合、权重柱状图 | 下拉切换风险档，指标和推荐语联动 |
| 🤖 AI助手 | 预制场景 + DeepSeek 实时问答 | 预制场景无需 API Key 即可体验 |
| 🛡️ 风险划分 | C1-C5 等级表、问卷概要、策略匹配 | 标注了 C1/C5 的回测覆盖范围 |

## 三、常见问题

**Q: 启动报错？**
A: 先跑 `pip install -r requirements.txt`，确保 pandas/plotly/streamlit 都装了。

**Q: 端口被占用？**
A: `streamlit run app.py --server.port 8502` 换一个端口。

**Q: DeepSeek 问答没反应？**
A: 需要填 API Key（在 DeepSeek 官网申请），不填的话切换到"预制场景"模式也能体验。

**Q: WSL 下运行浏览器不弹？**
A: WSL 不会自动弹浏览器，终端会显示 `Network URL`，复制到 Windows 浏览器打开即可。

## 四、数据文件

所有数据文件在 `data/` 目录下，由 A 的回测引擎生成。如需更新数据，替换对应 CSV 后重启 Streamlit 即可。

- `data/clean/market_score_daily.csv` — 评分数据（2798 天，含北向 NaN 修正 + 宏观滞后）
- `data/results/backtest/baseline_metrics.csv` — 8 策略对比表
- `data/results/backtest/baseline_nav.csv` — 日频净值序列
- `data/results/risk_tiers.csv` — 三档风险组合

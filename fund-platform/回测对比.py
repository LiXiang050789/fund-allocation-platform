import pandas as pd
import numpy as np
import os

# ===================== 路径配置 =====================
BASE_PATH = r"E:\农行杯\2026\clean"
PRICE_CSV = os.path.join(BASE_PATH, "etf_price_clean.csv")
SCORE_CSV = os.path.join(BASE_PATH, "market_score_daily.csv")
PPO_NET_CSV = os.path.join(BASE_PATH, "ppo_net_value.csv")
OUTPUT_CSV = os.path.join(BASE_PATH, "strategy_compare.csv")

# 1、加载数据
price = pd.read_csv(PRICE_CSV, encoding="utf-8-sig", parse_dates=["date"])
score = pd.read_csv(SCORE_CSV, encoding="utf-8-sig", parse_dates=["date"])
# 合并市场总分
df = price.merge(score[["date", "total_score"]], on="date", how="left")
df = df.sort_values("date").reset_index(drop=True)
assets = ["hs300", "zz500", "kc50", "consume", "chip", "gold", "bond10"]

# 计算每日收益率
ret_df = df[assets].pct_change().fillna(0)
total_days = len(df)


# 2、基准1：等权策略
def backtest_equal():
    weight = np.full(len(assets), 1 / len(assets))
    port_ret = ret_df @ weight
    return (1 + port_ret).cumprod()


# 3、基准2：20日动量策略
def backtest_momentum(window=20):
    momentum_ret_list = []
    for i in range(total_days):
        if i < window:
            momentum_ret_list.append(0.0)
            continue
        # 过去window日累计收益
        past_cum = ret_df.iloc[i - window:i].sum()
        # 选累计收益前3资产
        top3_asset = past_cum.nlargest(3).index.tolist()
        w = np.zeros(len(assets))
        # 权益各0.2，债券固定0.4
        for asset in top3_asset:
            idx = assets.index(asset)
            w[idx] = 0.2
        w[assets.index("bond10")] = 0.4
        # 权重归一化，保证总和=1
        w = w / np.sum(w)
        daily_r = ret_df.iloc[i] @ w
        momentum_ret_list.append(daily_r)
    # 累计净值
    return (1 + pd.Series(momentum_ret_list)).cumprod()


# 4、基准3：估值价值策略（基于market_score）
def backtest_value():
    value_ret_list = []
    eq_list = ["hs300", "zz500", "kc50", "consume", "chip"]
    gold_idx = assets.index("gold")
    bond_idx = assets.index("bond10")
    eq_num = len(eq_list)
    for i in range(total_days):
        ts = df.loc[i, "total_score"]
        w = np.zeros(len(assets))
        if ts >= 60:
            eq_w = 0.6 / eq_num
            gld_w = 0.2
            bnd_w = 0.2
        elif ts >= 45:
            eq_w = 0.4 / eq_num
            gld_w = 0.3
            bnd_w = 0.3
        else:
            eq_w = 0.2 / eq_num
            gld_w = 0.2
            bnd_w = 0.6
        # 分配权益权重
        for a in eq_list:
            w[assets.index(a)] = eq_w
        w[gold_idx] = gld_w
        w[bond_idx] = bnd_w
        w = w / np.sum(w)
        daily_r = ret_df.iloc[i] @ w
        value_ret_list.append(daily_r)
    return (1 + pd.Series(value_ret_list)).cumprod()


# 5、指标计算函数（修复年化收益bug、增加保护）
def calc_metrics(net_series):
    series = net_series.copy()
    total_day = len(series)
    if total_day <= 1:
        return {"年化收益": 0.0, "最大回撤": 0.0, "夏普比率": 0.0}

    # 复利年化收益（正确公式）
    final_net = series.iloc[-1]
    annual_return = (final_net ** (252 / total_day)) - 1

    # 最大回撤（输出负数，行业标准）
    peak = series.iloc[0]
    max_drawdown = 0.0
    for val in series:
        peak = max(peak, val)
        drawdown = (val - peak) / peak
        max_drawdown = min(max_drawdown, drawdown)

    # 夏普比率，除零保护
    daily_ret = series.diff().fillna(0)
    mean_dr = daily_ret.mean()
    std_dr = daily_ret.std()
    if std_dr < 1e-8:
        sharpe = 0.0
    else:
        sharpe = mean_dr / std_dr * np.sqrt(252)

    return {
        "年化收益": round(annual_return, 4),
        "最大回撤": round(max_drawdown, 4),
        "夏普比率": round(sharpe, 4)
    }


# ===================== 运行所有基准策略 =====================
print("开始计算基准策略曲线...")
eq_curve = backtest_equal()
mom_curve = backtest_momentum(window=20)
val_curve = backtest_value()

# 加载PPO训练净值，文件不存在则提示跳过
ppo_curve = None
if os.path.exists(PPO_NET_CSV):
    ppo_df = pd.read_csv(PPO_NET_CSV, encoding="utf-8-sig")
    ppo_curve = ppo_df["net_value"].reset_index(drop=True)
    # 长度对齐
    if len(ppo_curve) > total_days:
        ppo_curve = ppo_curve.iloc[:total_days]
    elif len(ppo_curve) < total_days:
        pad = [ppo_curve.iloc[-1]] * (total_days - len(ppo_curve))
        ppo_curve = pd.concat([ppo_curve, pd.Series(pad)], ignore_index=True)
else:
    print(f"警告：PPO净值文件不存在 {PPO_NET_CSV}，跳过PPO对比")

# ===================== 指标打印对比 =====================
print("\n===== 策略回测指标对比 =====")
metrics_eq = calc_metrics(eq_curve)
print(f"等权策略：{metrics_eq}")

metrics_mom = calc_metrics(mom_curve)
print(f"20日动量策略：{metrics_mom}")

metrics_val = calc_metrics(val_curve)
print(f"估值价值策略：{metrics_val}")

if ppo_curve is not None:
    metrics_ppo = calc_metrics(ppo_curve)
    print(f"PPO强化学习：{metrics_ppo}")

# ===================== 保存曲线文件 =====================
save_dict = {
    "date": df["date"],
    "等权": eq_curve,
    "动量": mom_curve,
    "价值": val_curve
}
if ppo_curve is not None:
    save_dict["PPO"] = ppo_curve

res_df = pd.DataFrame(save_dict)
res_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n完整净值曲线已保存至：{OUTPUT_CSV}")

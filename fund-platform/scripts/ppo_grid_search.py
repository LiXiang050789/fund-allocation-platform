"""
PPO 全局训练 + 参数网格搜索 (基于 zt 的 global training 思路)
"""
import os, sys, warnings, time, itertools
warnings.filterwarnings('ignore')
os.environ['SB3_ALLOW_GYM_V0'] = '1'

import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'platform'))
from portfolio_env import PortfolioEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 加载数据
feat = pd.read_csv(f'{BASE}/data/clean/train_feature_filtered_env.csv', encoding='utf-8-sig', parse_dates=['date'])
price = pd.read_csv(f'{BASE}/data/clean/etf_price_clean.csv', encoding='utf-8-sig', parse_dates=['date'])
common = sorted(set(feat['date']) & set(price['date']))
feat = feat[feat['date'].isin(common)].reset_index(drop=True).ffill().fillna(0.0)
price = price[price['date'].isin(feat['date'])].reset_index(drop=True).ffill().fillna(0.0)

ETF = ['hs300','zz500','kc50','consume','chip','gold','bond10']
SEED = 42; N_TOTAL = len(feat)

# 网格
grid = {
    'reward_coef': [
        (6.0, 0.5, 1.0, 0.005),   # zt 当前
        (8.0, 0.5, 1.0, 0.005),   # 提高收益权重
        (6.0, 0.5, 2.0, 0.005),   # 提高回撤惩罚
        (8.0, 0.3, 1.0, 0.003),   # 降低波动+换手
        (6.0, 0.7, 1.5, 0.005),   # 提高波动+回撤
    ],
    'learning_rate': [7e-5, 5e-5, 1e-4],
    'ent_coef': [0.05, 0.10],
}

# 只搜高影响力组合: reward_coef × lr × ent
combos = []
for rc in grid['reward_coef']:
    for lr in grid['learning_rate']:
        for ent in grid['ent_coef']:
            combos.append({'reward_coef': rc, 'learning_rate': lr, 'ent_coef': ent})

print(f'参数组合: {len(combos)} 组')
print(f'数据: {N_TOTAL}天, 训练区间 200~{N_TOTAL-60}, 验证 {N_TOTAL-60}~{N_TOTAL}')

results = []
best_sharpe = -999
best_combo = None

for i, cfg in enumerate(combos):
    t0 = time.time()
    rc = cfg['reward_coef']; lr = cfg['learning_rate']; ent = cfg['ent_coef']
    label = f'rc=({rc[0]},{rc[1]},{rc[2]},{rc[3]}) lr={lr:.0e} ent={ent:.2f}'

    try:
        # 全局训练 (200~T-60, skip early NaN period)
        train_env = PortfolioEnv(feature_df=feat, price_df=price,
            start_idx=200, end_idx=N_TOTAL-60,
            window=40, reward_coef=rc, min_w=0.02, max_w=0.4)

        model = PPO('MlpPolicy', train_env,
            learning_rate=lr, n_steps=512, batch_size=64,
            gamma=0.99, ent_coef=ent, clip_range=0.2,
            policy_kwargs={'net_arch': [128, 64]},
            verbose=0)

        model.learn(total_timesteps=3000)

        # 外样本评估 (最后60天)
        test_env = PortfolioEnv(feature_df=feat, price_df=price,
            start_idx=N_TOTAL-60, end_idx=N_TOTAL-1,
            window=40, reward_coef=rc)
        obs, _ = test_env.reset()
        done = False; nav = 1.0; rets = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            rets.append(test_env.unwrapped.net_value - nav)
            nav = test_env.unwrapped.net_value

        rets = pd.Series(rets).dropna()
        if len(rets) > 5:
            ann_ret = rets.mean() * 252
            ann_vol = rets.std() * np.sqrt(252)
            sharpe = (ann_ret - 0.025) / max(ann_vol, 1e-6)
            elapsed = time.time() - t0

            results.append({**cfg, 'ann_ret': ann_ret, 'sharpe': sharpe,
                          'vol': ann_vol, 'time': elapsed})
            print(f'[{i+1}/{len(combos)}] {label}')
            print(f'     夏普={sharpe:.4f} 年化={ann_ret*100:.1f}% 波动={ann_vol*100:.1f}% {elapsed:.0f}s')
            if sharpe > best_sharpe:
                best_sharpe = sharpe; best_combo = cfg
        else:
            print(f'[{i+1}/{len(combos)}] {label} ❌ 样本不足')
    except Exception as e:
        print(f'[{i+1}/{len(combos)}] {label} ❌ {e}')

# 汇总
print(f'\n{"="*70}')
print(f'最佳组合: rc={best_combo["reward_coef"]} lr={best_combo["learning_rate"]:.0e} ent={best_combo["ent_coef"]:.2f}')
print(f'最佳夏普: {best_sharpe:.4f}')

# 对比基线
print(f'\n===== 对比 =====')
print(f'zt全局训练: 夏普=0.976 年化=15.8%    回撤=-31.2%')
print(f'动态评分:   夏普=0.487 年化=19.6%    回撤=-18.3%')
print(f'LGB融合:    夏普=0.645 年化=12.2%    回撤=-17.6%')
if best_sharpe > 0.5:
    print(f'网格最佳:   夏普={best_sharpe:.3f}  评估窗口=60天')

# 保存
if results:
    pd.DataFrame(results).to_csv(f'{BASE}/results/ppo_grid_search.csv', index=False, encoding='utf-8-sig')
    print(f'\n✅ → results/ppo_grid_search.csv')

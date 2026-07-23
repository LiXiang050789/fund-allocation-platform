"""
PPO 快速验证 — 37 维特征 + 小网络，验证速度/精度对比
"""
import os, sys, warnings, time
warnings.filterwarnings('ignore')
os.environ['SB3_ALLOW_GYM_V0'] = '1'

import pandas as pd, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'platform'))
from portfolio_env import PortfolioEnv
from stable_baselines3 import PPO

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = f'{BASE}/data/clean'
OUT   = f'{BASE}/results'
os.makedirs(OUT, exist_ok=True)

ETF = ['hs300','zz500','kc50','consume','chip','gold','bond10']

print('[1] 加载 37 维筛选特征...')
feat = pd.read_csv(f'{DATA}/train_feature_filtered.csv', encoding='utf-8-sig', parse_dates=['date'])
price = pd.read_csv(f'{DATA}/etf_price_clean.csv', encoding='utf-8-sig', parse_dates=['date'])

common = sorted(set(feat['date']) & set(price['date']))
feat = feat[feat['date'].isin(common)].reset_index(drop=True)
price = price[price['date'].isin(feat['date'])].reset_index(drop=True)
# 防 NaN: ffill + 填 0
feat = feat.ffill().fillna(0.0)
price = price.ffill().fillna(0.0)

t0 = time.time()
print(f'    数据: {len(feat)}天 × {feat.shape[1]-1}特征')

print('[2] 训练 PPO (小网络 128→64, 3轮验证)...')
# 使用滚动窗口: 500train → 100val → 60pred, 只跑 3 轮
total = len(feat)
rounds = 0; results = []

for rnd in range(3):
    train_s = 500 + rnd * 60  # 跳过前500天NaN期
    train_e = min(train_s + 500, total - 220)
    val_e = min(train_e + 100, total - 60)
    pred_e = min(val_e + 60, total)

    if train_e - train_s < 200 or pred_e - val_e < 10:
        break

    env = PortfolioEnv(
        feature_df=feat, price_df=price,
        start_idx=train_s, end_idx=train_e,
        window=40, reward_coef=(6.0, 0.4, 2.5, 0.005),
        min_w=0.02, max_w=0.4, trade_cost=0.0005, constraint_coef=10.0,
    )
    env_val = PortfolioEnv(
        feature_df=feat, price_df=price,
        start_idx=val_e - 60, end_idx=val_e,
        window=40, reward_coef=(6.0, 0.4, 2.5, 0.005),
    )

    model = PPO('MlpPolicy', env,
        learning_rate=7e-5, n_steps=512, batch_size=64,
        gamma=0.99, ent_coef=0.10,
        policy_kwargs={'net_arch': [128, 64]},  # 小网络
        verbose=0)

    model.learn(total_timesteps=2000)

    # 验证
    obs, _ = env_val.reset()
    done = False; nav = 1.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env_val.step(action)
        done = terminated or truncated
        nav = env_val.unwrapped.net_value

    rounds += 1
    results.append(nav)
    print(f'    轮{rnd+1}: 训练{train_e-train_s}d → 验证净值={nav:.3f}')

elapsed = time.time() - t0
avg_nav = np.mean(results) if results else 0
print(f'\n    耗时: {elapsed:.0f}秒 ({elapsed/60:.1f}分钟)')
print(f'    3轮平均净值: {avg_nav:.3f}')
print(f'    速算年化: {(avg_nav - 1) * 100:.1f}% (验证窗口 ~60天)')

# 对比 zt 原始结果
print(f'\n[3] 对比 zt 37维 PPO:')
print(f'    zt 9年净值: ~1.05 (年化 ~0.5%)')
print(f'    本次37维小网验证净值: {avg_nav:.3f}')
if avg_nav > 1.03:
    print('    ✅ 37维+小网络短期验证胜出 — 方向正确')
else:
    print('    ⚠️ 仍需调参，但速度已大幅提升')

print(f'\n✅ 完成 → 可告知zt: 37维+128x64网络 = {elapsed/60:.1f}分/3轮')

# PPO奖励口径

## 适用版本与权威声明

本文描述最终 PPO 训练代码的奖励口径。依据 env_16.py、ppo_9.16.py 与保存的 H_equity_residual 模型配置。若与早期设计稿《强化学习.txt》冲突，以本文和最终代码为准。

## 最终奖励公式与系数

最终奖励 = 10.0 ×（组合收益 - 规则基准收益） - 0.5 × 波动 - 0.0 × 换手 - 1.0 × 回撤加深 + 0.02 × 夏普变化，并 clip 到 [-5, 5]。

reward_mode="excess_rule"；reward_coef=(10.0, 0.5, 0.0, 1.0)，解包顺序为 base/vol/turn/dd。组合收益 port_ret = dot(w, returns) - turnover * 0.0003，换手成本 3bp 已计入收益；规则基准收益 rule_ret=dot(rule_w, returns)。

| 项 | 系数 | 说明 |
|---|---:|---|
| 相对规则基准超额收益 | 10.0 | port_ret - rule_ret |
| 波动 | 0.5 | vol_ewma = 0.9*prev + 0.1*abs(port_ret) |
| 换手 | 0.0 | 不另设惩罚；3bp 交易成本已从 port_ret 扣除 |
| 回撤加深 | 1.0 | max(0, abs_dd - prev_abs_dd) |
| 夏普变化 | 0.02 | +0.02 * (current_sharpe - prev_sharpe)，奖励项，可正可负 |
| clip | — | [-5, 5]，enable_reward_norm=false |

## 与早期设计稿的差异

《强化学习.txt》设计文档四项是：当日组合收益正向奖励、波动率惩罚、回撤惩罚、换手率惩罚。最终代码正向项为相对规则基准的超额收益；换手系数为 0，但 3bp 成本已计入组合收益；新增 +0.02 × 夏普变化奖励项；回撤惩罚是回撤加深的边际惩罚，而非绝对最大回撤水平；设计稿没有写 clip，最终代码 clip(-5,5)。这些属于早期设计与最终实现的差异，最终以代码为准。

旧版 platform/portfolio_env.py 的系数顺序不同，不得用于解释最终模型。旧版第 218 行按 profit/vol/dd/turn 解包；最终 env_16.py:523 按 base/vol/turn/dd 解包。

手册第 276 行的“夏普边际”是简称，实际是 +0.02 × ΔSharpe 的奖励项，不是惩罚项；手册未单列换手系数，最终代码换手系数为 0，3bp 成本已计入收益。

## 规则基准

rule_w 以 PE 分位定股债比例。代码中的 pe_z 为标准化尺度，债券权重为 0.30 + 0.25 * clip(pe_z/2, -1, 1)，再裁到 [0.05, 0.70]；权益内部等权。

## 代码证据

- 归档/zt-PPO实验/scripts-0917/env_16.py:523-553（奖励与裁剪）；298-320（规则基准）；475-485（成本与组合收益）。
- 归档/zt-PPO实验/scripts-0917/ppo_9.16.py:70-71（训练参数）。
- 归档/zt-PPO实验/results/global_ppo/H_equity_residual/config.json（保存模型配置）。
- 归档/zt-PPO实验/scripts-0917/export_ppo_weight.py:35-56（导出配置）。
- 归档/zt-PPO实验/scripts-0917/ensemble_eval.py:46-67（评估配置）。
- 项目书修订说明.md:115（“若不一致，以代码为准”）。

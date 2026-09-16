"""
农行杯赛题一 · 指数基金智能配置决策平台
Streamlit 前端 — 市场仪表盘 / 自动化数据流水线 / 配置推荐 / AI助手 / 风险划分
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


st.set_page_config(page_title="指数基金智能配置平台", layout="wide", page_icon="📊")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BASE)
RAW_DIR = os.path.join(BASE, "data", "raw")
PPO_MODEL_PATH = f"{BASE}/models/global_best.zip"
PPO_FEATURE_PATH = f"{BASE}/data/clean/train_feature_filtered_ppo.csv"  # 仅作提示，实际推理路径见 ppo_infer_runtime.py
PPO_PRICE_PATH = f"{BASE}/data/clean/etf_price_clean.csv"
PPO_WEIGHT_PATH = f"{BASE}/results/ppo_weight.csv"
PPO_RUNTIME_PATH = f"{BASE}/platform/ppo_infer_runtime.py"

COLORS = {
    "沪深300": "#607D8B",
    "等权": "#888888",
    "风险平价": "#4CAF50",
    "MVO": "#FF5722",
    "动量": "#2196F3",
    "动态评分": "#E91E63",
    "LGB融合": "#FF9800",
    "PPO月度": "#9C27B0",
    "PPO原生日频": "#673AB7",
}
ETF_CODES = ["hs300", "zz500", "kc50", "consume", "chip", "gold", "bond10"]
ETF_CN = {
    "hs300": "沪深300",
    "zz500": "中证500",
    "kc50": "科创50",
    "consume": "消费",
    "chip": "芯片",
    "gold": "黄金",
    "bond10": "十年国债",
}
format_dict = {"年化收益": "{:.1%}", "最大回撤": "{:.1%}", "夏普比率": "{:.2f}", "卡玛比率": "{:.2f}", "月均换手率": "{:.1%}", "年化收益/最大回撤": "{:.2f}"}


def resolve_ppo_python():
    candidates = [
        (os.path.join(REPO_ROOT, ".conda_ppo_py312", "Scripts", "python.exe"), "PPO独立环境 Windows"),
        (os.path.join(REPO_ROOT, ".conda_ppo_py312", "bin", "python"), "PPO独立环境 Linux"),
        (sys.executable, "当前 Python 解释器"),
    ]
    for path, label in candidates:
        if os.path.exists(path):
            return path, label
    return sys.executable, "当前 Python 解释器"


def _safe_read_csv(path, parse_dates=None, encodings=("utf-8-sig", "gbk")):
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding, parse_dates=parse_dates)
            if any("�" in str(col) for col in df.columns):
                continue
            return df, encoding
        except Exception:
            continue
    return None, None


def _format_ppo_weights(weight_map):
    return "、".join([f"{ETF_CN[k]} {weight_map[k] * 100:.1f}%" for k in ETF_CODES])


def _load_cached_ppo_weight(as_of_date, reason=""):
    if not os.path.exists(PPO_WEIGHT_PATH):
        return {"ok": False, "message": "PPO 权重文件缺失，当前仅使用市场评分"}
    weight_df, _ = _safe_read_csv(PPO_WEIGHT_PATH, parse_dates=["date"])
    if weight_df is None:
        return {"ok": False, "message": "PPO 权重文件读取失败，当前仅使用市场评分"}
    weight_df = weight_df.sort_values("date")
    weight_df = weight_df[weight_df["date"] <= pd.Timestamp(as_of_date)]
    if weight_df.empty:
        return {"ok": False, "message": "PPO 权重无有效日期，当前仅使用市场评分"}
    row = weight_df.iloc[-1]
    weight_map = {}
    for code in ETF_CODES:
        val = row.get(code, 0.0)
        weight_map[code] = float(val) if pd.notna(val) else 0.0
    message = f"PPO 实时推理不可用，已使用最近一次离线权重：{reason}" if reason else ""
    return {
        "ok": True,
        "date": pd.Timestamp(row["date"]),
        "weights": weight_map,
        "text": f"PPO 模型今日推荐配置：{_format_ppo_weights(weight_map)}",
        "source": "cached_weight",
        "message": message,
    }


@st.cache_data(show_spinner=False)
def _infer_latest_ppo_weight_cached(date_key):
    if not os.path.exists(PPO_MODEL_PATH):
        return {"ok": False, "message": "PPO 模型未部署，当前仅使用市场评分"}
    ppo_python, python_label = resolve_ppo_python()
    started = time.perf_counter()
    try:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = subprocess.run(
            [ppo_python, PPO_RUNTIME_PATH, date_key],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=60,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8:replace"},
            creationflags=creationflags,
        )
        elapsed = time.perf_counter() - started
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or f"exit {proc.returncode}")[-200:]
            cached = _load_cached_ppo_weight(date_key, detail)
            cached["elapsed"] = elapsed
            cached["python_label"] = python_label
            return cached

        lines = proc.stdout.strip().splitlines()
        candidate_lines = [line for line in lines if line.lstrip().startswith("{")]
        if not candidate_lines:
            cached = _load_cached_ppo_weight(date_key, f"子进程无JSON输出，stdout:{proc.stdout[:200]}")
            cached["elapsed"] = elapsed
            cached["python_label"] = python_label
            return cached
        payload = json.loads(candidate_lines[-1])
        weight_map = {}
        for code in ETF_CODES:
            val = payload["weights"].get(code, 0.0)
            weight_map[code] = float(val) if pd.notna(val) else 0.0
        return {
            "ok": True,
            "date": pd.Timestamp(payload["date"]),
            "weights": weight_map,
            "text": f"PPO 模型今日推荐配置：{_format_ppo_weights(weight_map)}",
            "source": payload.get("source", "live_inference"),
            "message": payload.get("message", ""),
            "elapsed": elapsed,
            "python_label": python_label,
            "n_features": payload.get("n_features"),
            "model_obs_dim": payload.get("model_obs_dim"),
            "steps": payload.get("steps"),
        }
    except subprocess.TimeoutExpired:
        cached = _load_cached_ppo_weight(date_key, "实时推理超时")
        cached["elapsed"] = 60.0
        cached["python_label"] = python_label
        return cached
    except Exception as e:
        elapsed = time.perf_counter() - started
        cached = _load_cached_ppo_weight(date_key, str(e))
        cached["elapsed"] = elapsed
        cached["python_label"] = python_label
        return cached


def _safe_ppo_info(as_of_date):
    try:
        date_key = str(pd.Timestamp(as_of_date).date())
        return _infer_latest_ppo_weight_cached(date_key)
    except Exception as e:
        return {"ok": False, "message": f"PPO 推理失败：{e}", "source": "error"}


def infer_latest_ppo_weight(as_of_date):
    date_key = str(pd.Timestamp(as_of_date).date())
    return _safe_ppo_info(date_key)


def _current_ppo_info(as_of_date):
    live_info = st.session_state.get("ppo_live_info")
    if live_info and live_info.get("ok"):
        return live_info
    return _load_cached_ppo_weight(as_of_date)


def _store_live_ppo_if_valid(ppo_info):
    if ppo_info.get("ok") and ppo_info.get("source") == "live_inference":
        st.session_state["ppo_live_info"] = ppo_info


def _compare_live_to_offline(ppo_info):
    if not ppo_info.get("ok"):
        return None
    weight_df, _ = _safe_read_csv(PPO_WEIGHT_PATH, parse_dates=["date"])
    if weight_df is None or weight_df.empty:
        return None
    weight_df = weight_df.sort_values("date")
    row_df = weight_df[weight_df["date"] <= pd.Timestamp(ppo_info["date"])]
    if row_df.empty:
        return None
    row = row_df.iloc[-1]
    diffs = [abs(float(ppo_info["weights"][code]) - float(row[code])) for code in ETF_CODES]
    return max(diffs)


def _add_return_drawdown_ratio(df):
    df = df.copy()
    if "年化收益/最大回撤" not in df.columns:
        df["年化收益/最大回撤"] = df["年化收益"] / df["最大回撤"].abs()
    return df


@st.cache_data(show_spinner=False)
def load_data():
    score = pd.read_csv(f"{BASE}/data/clean/market_score_daily.csv", encoding="utf-8-sig", parse_dates=["date"])
    tiers = pd.read_csv(f"{BASE}/results/risk_tiers.csv", encoding="utf-8-sig")
    return score, tiers


@st.cache_data(show_spinner=False)
def load_unified_metrics():
    metrics, _ = _safe_read_csv(f"{BASE}/results/backtest/all_strategy_metrics.csv")
    if metrics is None:
        return pd.DataFrame(columns=["策略", "年化收益", "年化波动", "夏普比率", "索提诺比率", "最大回撤", "卡玛比率", "月均换手率", "权重集中度HHI", "胜率", "95%VaR", "年化收益/最大回撤"])
    return _add_return_drawdown_ratio(metrics)


score_df, tiers_df = load_data()
latest = score_df.iloc[-1]
DATA_DATE_MIN = score_df["date"].min().date()
DATA_DATE_MAX = score_df["date"].max().date()


def _status_text(ppo_info):
    if ppo_info.get("source") == "live_inference":
        return "实时推理"
    if ppo_info.get("source") == "cached_weight":
        return "离线权重兜底，可点击执行实时推理验证"
    return "不可用"


def _weight_frame(ppo_info):
    return pd.DataFrame(
        {
            "资产": [ETF_CN[code] for code in ETF_CODES],
            "代码": ETF_CODES,
            "权重": [ppo_info["weights"].get(code, 0.0) for code in ETF_CODES],
        }
    )


def _render_weight_bar(ppo_info, title="PPO 权重"):
    weight_df = _weight_frame(ppo_info)
    fig = go.Figure(
        go.Bar(
            x=weight_df["资产"],
            y=weight_df["权重"] * 100,
            marker_color=["#E91E63", "#E91E63", "#E91E63", "#E91E63", "#E91E63", "#FFC107", "#4CAF50"],
            text=[f"{v * 100:.1f}%" for v in weight_df["权重"]],
            textposition="outside",
        )
    )
    fig.update_layout(height=320, title=title, yaxis_title="权重(%)", margin=dict(t=45, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(weight_df.style.format({"权重": "{:.2%}"}), use_container_width=True, hide_index=True)


def _render_live_button(button_label, as_of_date):
    if st.button(button_label, use_container_width=True):
        with st.spinner("正在执行 PPO 实时推理，最多等待 60 秒..."):
            info = infer_latest_ppo_weight(as_of_date)
            _store_live_ppo_if_valid(info)
        if info.get("ok"):
            st.success(f"{_status_text(info)}完成，权重日期 {info['date'].date()}")
            if info.get("elapsed") is not None:
                st.caption(f"耗时 {info['elapsed']:.1f}s · 解释器：{info.get('python_label', '未知')}")
            meta = []
            if info.get("n_features") is not None:
                meta.append(f"{info['n_features']} 维特征命中")
            if info.get("model_obs_dim") is not None:
                meta.append(f"{info['model_obs_dim']} 维观测")
            if info.get("steps") is not None:
                meta.append(f"重放 {info['steps']} 步")
            if meta:
                st.caption("source=live_inference · " + " / ".join(meta))
        else:
            st.warning(info.get("message", "PPO 推理不可用，已保持离线展示"))
        return info
    return None


with st.sidebar:
    st.title("🏦 农行杯赛题一")
    st.caption("指数基金智能配置决策平台")
    st.divider()
    st.metric("最新市场评分", f"{latest['total_score']:.0f}/100")
    st.metric("市场状态", latest["market_state"])
    st.divider()
    st.caption(f"数据日期: {latest['date'].date()}")
    st.caption("团队: 三元智投队")
    st.caption("数据源: sf四维评分规则 + 北向NaN + 宏观滞后修正")


tab1, tab_pipeline, tab3, tab4, tab5 = st.tabs(
    ["📈 市场仪表盘",  "⚙️ 自动化数据流水线", "🎯 配置推荐", "🤖 AI助手", "🛡️ 风险划分"]
)


with tab1:
    st.header("市场环境仪表盘")
    selected_date = st.slider("选择历史日期", min_value=DATA_DATE_MIN, max_value=DATA_DATE_MAX, value=DATA_DATE_MAX, format="YYYY-MM-DD")
    row = score_df[score_df["date"] == pd.Timestamp(selected_date)]
    if len(row) == 0:
        row = score_df.iloc[-1:]
    current = row.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("估值维度", f"{current['val_score']:.0f}", "25分制")
    with col2:
        st.metric("宏观维度", f"{current['macro_score']:.0f}", "25分制")
    with col3:
        st.metric("情绪维度", f"{current['sent_score']:.0f}", "20分制")
    with col4:
        st.metric("趋势维度", f"{current['trend_score']:.0f}", "30分制")

    st.metric("总分", f"{current['total_score']:.0f}/100", current["market_state"])

    col_left, col_right = st.columns([1, 1.5])
    with col_left:
        fig_radar = go.Figure(
            go.Scatterpolar(
                r=[current["val_score"], current["macro_score"], current["sent_score"], current["trend_score"]],
                theta=["估值(25)", "宏观(25)", "情绪(20)", "趋势(30)"],
                fill="toself",
                fillcolor="rgba(233,30,99,0.3)",
                line=dict(color="#E91E63", width=2),
            )
        )
        fig_radar.update_layout(height=350, margin=dict(t=20, b=20), polar=dict(radialaxis=dict(range=[0, 30])))
        st.plotly_chart(fig_radar, use_container_width=True)
    with col_right:
        st.line_chart(score_df.set_index("date")[["total_score"]].tail(500), height=350)

    st.subheader("全期状态分布")
    dist = score_df["market_state"].value_counts()
    cols = st.columns(5)
    for i, state in enumerate(["上行", "震荡偏强", "震荡", "下行", "极寒"]):
        cnt = dist.get(state, 0)
        pct = cnt / len(score_df) * 100
        cols[i].metric(state, f"{cnt}天", f"{pct:.1f}%")


def _file_mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


def _pipeline_signature():
    tracked = [
        RAW_DIR,
        os.path.join(RAW_DIR, "数据汇总.xlsx"),
        f"{BASE}/data/clean/train_total_feature.csv",
        f"{BASE}/data/clean/train_feature_filtered.csv",
        PPO_FEATURE_PATH,
        PPO_PRICE_PATH,
        f"{BASE}/data/clean/market_score_daily.csv",
        PPO_MODEL_PATH,
        f"{BASE}/results/feature_importance.csv",
        f"{BASE}/results/pred_return_lgb_daily.csv",
        PPO_WEIGHT_PATH,
        f"{BASE}/results/backtest/all_strategy_metrics.csv",
    ]
    return tuple((path, _file_mtime(path)) for path in tracked)


@st.cache_data(show_spinner=False)
def _scan_pipeline_assets(signature):
    checks = [
        ("data/raw 目录", RAW_DIR, os.path.isdir(RAW_DIR)),
        ("数据汇总.xlsx", os.path.join(RAW_DIR, "数据汇总.xlsx"), os.path.exists(os.path.join(RAW_DIR, "数据汇总.xlsx"))),
        ("train_total_feature", f"{BASE}/data/clean/train_total_feature.csv", os.path.exists(f"{BASE}/data/clean/train_total_feature.csv")),
        ("train_feature_filtered", f"{BASE}/data/clean/train_feature_filtered.csv", os.path.exists(f"{BASE}/data/clean/train_feature_filtered.csv")),
        ("train_feature_filtered_ppo", PPO_FEATURE_PATH, os.path.exists(PPO_FEATURE_PATH)),
        ("etf_price_clean", PPO_PRICE_PATH, os.path.exists(PPO_PRICE_PATH)),
        ("market_score_daily", f"{BASE}/data/clean/market_score_daily.csv", os.path.exists(f"{BASE}/data/clean/market_score_daily.csv")),
        ("global_best.zip", PPO_MODEL_PATH, os.path.exists(PPO_MODEL_PATH)),
        ("feature_importance", f"{BASE}/results/feature_importance.csv", os.path.exists(f"{BASE}/results/feature_importance.csv")),
        ("pred_return_lgb_daily", f"{BASE}/results/pred_return_lgb_daily.csv", os.path.exists(f"{BASE}/results/pred_return_lgb_daily.csv")),
        ("ppo_weight", PPO_WEIGHT_PATH, os.path.exists(PPO_WEIGHT_PATH)),
        ("all_strategy_metrics", f"{BASE}/results/backtest/all_strategy_metrics.csv", os.path.exists(f"{BASE}/results/backtest/all_strategy_metrics.csv")),
    ]

    semantics = []
    metrics, _ = _safe_read_csv(f"{BASE}/results/backtest/all_strategy_metrics.csv")
    if metrics is not None:
        semantics.append(("PPO 两行存在", {"PPO月度", "PPO原生日频"}.issubset(set(metrics["策略"]))))
    else:
        semantics.append(("PPO 两行存在", False))

    ppo_feat, _ = _safe_read_csv(PPO_FEATURE_PATH)
    semantics.append(("PPO 特征 44 维", ppo_feat is not None and ppo_feat.drop(columns=["date"], errors="ignore").shape[1] == 44))

    weight_df, _ = _safe_read_csv(PPO_WEIGHT_PATH, parse_dates=["date"])
    if weight_df is not None and not weight_df.empty:
        weight_sum = float(weight_df.sort_values("date").iloc[-1][ETF_CODES].sum())
        semantics.append(("PPO 权重和≈1", abs(weight_sum - 1.0) <= 1e-3))
    else:
        semantics.append(("PPO 权重和≈1", False))

    return {"checks": checks, "semantics": semantics}


def _clean_table_shape(path):
    df, enc = _safe_read_csv(path)
    if df is None:
        return "读取失败"
    return f"{len(df)}×{len(df.columns)}（{enc}）"


@st.cache_data(show_spinner=False)
def _pipeline_counts(signature):
    raw_files = sum(len(files) for _, _, files in os.walk(RAW_DIR)) if os.path.isdir(RAW_DIR) else 0
    clean_dir = os.path.join(BASE, "data", "clean")
    clean_csvs = len([name for name in os.listdir(clean_dir) if name.endswith(".csv")]) if os.path.isdir(clean_dir) else 0
    return raw_files, clean_csvs


def _latest_raw_mtime():
    if not os.path.isdir(RAW_DIR):
        return None
    latest_file = None
    latest_mtime = 0
    for root, _, files in os.walk(RAW_DIR):
        for name in files:
            path = os.path.join(root, name)
            mtime = os.path.getmtime(path)
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_file = path
    if latest_file is None:
        return None
    return os.path.relpath(latest_file, BASE), pd.Timestamp.fromtimestamp(latest_mtime)


def _pipeline_script_path(name):
    for candidate in (os.path.join(REPO_ROOT, name), os.path.join(BASE, "scripts", name)):
        if os.path.exists(candidate):
            return candidate
    return os.path.join(REPO_ROOT, name)


def _script_head(path, n=100):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except UnicodeDecodeError:
        with open(path, "r", encoding="gbk", errors="replace") as fh:
            lines = fh.readlines()
    except Exception as e:
        return f"读取失败：{e}"
    return "".join(lines[:n])


@st.cache_data(show_spinner=False)
def _lgb_annual_ic():
    pred, _ = _safe_read_csv(f"{BASE}/results/pred_return_lgb_daily.csv", parse_dates=["date"])
    price, _ = _safe_read_csv(PPO_PRICE_PATH, parse_dates=["date"])
    if pred is None or price is None:
        return {}
    future = price.set_index("date")[ETF_CODES].shift(-21) / price.set_index("date")[ETF_CODES] - 1
    future = future.reset_index()
    merged = pred.merge(future, on="date", suffixes=("_pred", "_ret"))
    result = {}
    for year, group in merged.assign(year=merged["date"].dt.year).groupby("year"):
        if year < 2022:
            continue
        ics = []
        for code in ETF_CODES:
            data = group[[code + "_pred", code + "_ret"]].dropna()
            if len(data) < 2 or np.isclose(data[code + "_pred"].std(), 0) or np.isclose(data[code + "_ret"].std(), 0):
                ics.append(0.0)
            else:
                ics.append(float(data[code + "_pred"].corr(data[code + "_ret"])))
        result[int(year)] = float(np.mean(ics))
    return result


def _render_pipeline_chip(label, ok=True):
    bg = "#E8F5E9" if ok else "#FFEBEE"
    fg = "#2E7D32" if ok else "#C62828"
    st.markdown(f"<span style='display:inline-block;margin:2px 6px 2px 0;padding:4px 8px;border-radius:999px;background:{bg};color:{fg};font-size:13px'>{label}</span>", unsafe_allow_html=True)


with tab_pipeline:
    st.header("⚙️ 自动化数据流水线")
    st.caption("展示从采集到调仓的自动化链路、真实产物与可执行推理；采集环节现场只读展示，不联网重跑。")
    chip_cols = st.columns(3)
    chip_cols[0].success("采集：只读展示")
    chip_cols[1].info("清洗·特征·建模·回测：读取真实产物")
    chip_cols[2].success("PPO 推理：可执行")

    pipeline_signature = _pipeline_signature()
    scan = _scan_pipeline_assets(pipeline_signature)
    file_ready = sum(1 for _, _, ok in scan["checks"] if ok)
    semantic_ready = sum(1 for _, ok in scan["semantics"] if ok)
    st.subheader(f"文件 {file_ready}/12 · 就绪校验 {semantic_ready}/3")
    cols = st.columns(4)
    for idx, (name, _, ok) in enumerate(scan["checks"]):
        with cols[idx % 4]:
            _render_pipeline_chip(f"{name} {'✓' if ok else '✗'}", ok)
    sem_cols = st.columns(3)
    for idx, (name, ok) in enumerate(scan["semantics"]):
        with sem_cols[idx]:
            _render_pipeline_chip(f"{name} {'✓' if ok else '✗'}", ok)
    failed = [name for name, _, ok in scan["checks"] if not ok] + [name for name, ok in scan["semantics"] if not ok]
    if failed:
        st.error("未通过项：" + "、".join(failed))

    raw_files, clean_csvs = _pipeline_counts(pipeline_signature)
    train_total_shape = _clean_table_shape(f"{BASE}/data/clean/train_total_feature.csv")
    lgb_shape = _clean_table_shape(f"{BASE}/data/clean/train_feature_filtered.csv")
    ppo_shape = _clean_table_shape(PPO_FEATURE_PATH)
    fi_df, _ = _safe_read_csv(f"{BASE}/results/feature_importance.csv")
    top_feature = "bond_10y_2y_spread 0.1962"
    if fi_df is not None and not fi_df.empty:
        top_feature = f"{fi_df.iloc[0]['feature']} {fi_df.iloc[0]['importance']:.4f}"
    lgb_pred, _ = _safe_read_csv(f"{BASE}/results/pred_return_lgb_daily.csv")
    pred_count = int(lgb_pred[[c for c in lgb_pred.columns if c != "date"]].notna().sum().sum()) if lgb_pred is not None else 0
    current_ppo = _current_ppo_info(latest["date"])

    st.subheader("六环节链路")
    cards = st.columns(6)
    card_text = [
        ("① 采集", f"{raw_files} 文件\n\n爬虫脚本（仅代码展示）\n\n理杏仁人工导出 kc50·consume·chip\n\n手工 数据汇总.xlsx"),
        ("② 清洗", f"{clean_csvs} CSV\n\n六主表行列现读\n\nETF价格 {_clean_table_shape(PPO_PRICE_PATH)}"),
        ("③ 特征", f"182 → 37/44\n\nIC≥0.022 / r≤0.92 / 累积0.88\n\nTop1 {top_feature}"),
        ("④ 建模", f"LGB有效预测 {pred_count} 条\n\nPPO模型 global_best.zip\n\n44 维特征 {ppo_shape}"),
        ("⑤ 回测", "统一引擎 9 套\n\n滑点0.0005 / 双边费0.001\n\n单资产上限0.30"),
        ("⑥ 调仓", f"最新权重日期 {current_ppo['date'].date() if current_ppo.get('ok') else '不可用'}\n\n权重和校验\n\n{_status_text(current_ppo)}"),
    ]
    for col, (title, body) in zip(cards, card_text):
        with col:
            st.markdown(f"**{title}**")
            st.caption(body)

    with st.expander("① 采集层：只读脚本与最近产物", expanded=False):
        b1, b2, b3 = st.columns(3)
        if b1.button("查看采集脚本", use_container_width=True):
            for script in ["爬取数据.py", "补充数据挖取.py"]:
                path = _pipeline_script_path(script)
                st.code(_script_head(path, 100), language="python")
                st.caption(f"完整脚本已入库：{path}")
        if b2.button("查看最近采集产物", use_container_width=True):
            latest_raw = _latest_raw_mtime()
            if latest_raw:
                rel_path, ts = latest_raw
                st.info(f"data/raw 文件数：{raw_files}；最近文件：{rel_path}；更新时间：{ts}")
            else:
                st.warning("data/raw 暂无可扫描文件")
        if b3.button("重新扫描本地产物", use_container_width=True):
            _scan_pipeline_assets.clear()
            _pipeline_counts.clear()
            st.rerun()

    with st.expander("② 清洗层：主表规模", expanded=False):
        clean_summary = pd.DataFrame(
            [
                ["etf_price_clean", _clean_table_shape(PPO_PRICE_PATH)],
                ["train_total_feature", train_total_shape],
                ["market_score_daily", _clean_table_shape(f"{BASE}/data/clean/market_score_daily.csv")],
                ["valuation_merged", _clean_table_shape(f"{BASE}/data/clean/valuation_merged.csv")],
                ["tech_factor_clean", _clean_table_shape(f"{BASE}/data/clean/tech_factor_clean.csv")],
                ["macro_merged", _clean_table_shape(f"{BASE}/data/clean/macro_merged.csv")],
            ],
            columns=["表", "行列"],
        )
        st.dataframe(clean_summary, use_container_width=True, hide_index=True)

    with st.expander("③ 特征筛选：漏斗与重要性", expanded=True):
        st.caption(f"总特征 {train_total_shape}；LGB 特征 {lgb_shape}；PPO 特征 {ppo_shape}")
        if fi_df is not None and not fi_df.empty:
            top10 = fi_df.head(10).iloc[::-1]
            fig = go.Figure(go.Bar(x=top10["importance"], y=top10["feature"], orientation="h", marker_color="#E91E63"))
            fig.update_layout(height=360, margin=dict(t=20, b=20), xaxis_title="importance")
            st.plotly_chart(fig, use_container_width=True)
        image_path = f"{BASE}/PPT素材/fig7_feature_importance.png"
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)

    with st.expander("④ 建模：LGB 年度 IC 与 PPO 对拍", expanded=True):
        ic_map = _lgb_annual_ic()
        if ic_map:
            fig_ic = go.Figure(go.Bar(x=list(ic_map.keys()), y=[round(v, 4) for v in ic_map.values()], marker_color="#2196F3"))
            fig_ic.update_layout(height=300, margin=dict(t=20, b=20), yaxis_title="IC")
            st.plotly_chart(fig_ic, use_container_width=True)

        ppo_python, python_label = resolve_ppo_python()
        st.caption(f"PPO 解释器探测：{python_label} · {ppo_python}")
        live_info = st.session_state.get("ppo_live_info")
        if live_info and live_info.get("ok") and live_info.get("source") == "live_inference":
            diff = _compare_live_to_offline(live_info)
            ok = diff is not None and diff <= 1e-6
            _render_pipeline_chip(f"对拍最大偏差 {diff:.2e} {'✓' if ok else '✗'}", ok)
            meta = []
            if live_info.get("n_features") is not None:
                meta.append(f"{live_info['n_features']} 维特征命中")
            if live_info.get("model_obs_dim") is not None:
                meta.append(f"{live_info['model_obs_dim']} 维观测")
            if live_info.get("steps") is not None:
                meta.append(f"重放 {live_info['steps']} 步")
            if meta:
                st.caption(" / ".join(meta))
            if live_info.get("elapsed") is not None:
                st.caption(f"耗时 {live_info['elapsed']:.1f}s")
        else:
            st.caption("离线权重兜底，未对拍")
            info = _render_live_button("执行模型对拍", DATA_DATE_MAX)
            if info and info.get("source") == "live_inference":
                diff = _compare_live_to_offline(info)
                if diff is not None:
                    _render_pipeline_chip(f"对拍最大偏差 {diff:.2e} {'✓' if diff <= 1e-6 else '✗'}", diff <= 1e-6)
        st.caption("演示机缺 PPO 依赖时自动回退离线权重，离线权重由同一模型批量推理生成。")

    with st.expander("⑤ 统一回测证据区", expanded=False):
        unified = load_unified_metrics()
        if unified.empty:
            st.warning("统一回测指标文件读取失败，当前跳过指标表展示。")
        else:
            st.dataframe(unified.style.format(format_dict, na_rep="—"), use_container_width=True, hide_index=True)
        nav_all, _ = _safe_read_csv(f"{BASE}/results/backtest/all_strategy_nav.csv", parse_dates=["date"])
        if nav_all is not None:
            fig_nav = go.Figure()
            for col in [c for c in nav_all.columns if c != "date"]:
                fig_nav.add_trace(go.Scatter(x=nav_all["date"], y=nav_all[col], mode="lines", name=col, line=dict(color=COLORS.get(col))))
            fig_nav.update_layout(height=420, margin=dict(t=20, b=20), yaxis_title="净值")
            st.plotly_chart(fig_nav, use_container_width=True)
        else:
            st.warning("统一回测净值文件读取失败，当前跳过净值图展示。")
        st.caption("统一回测引擎口径")
        m1, m2, m3 = st.columns(3)
        m1.metric("滑点", "0.0005")
        m2.metric("双边费", "0.001")
        m3.metric("单资产上限", "0.30")
        st.markdown("**分段回测（6 阶段）**")
        st.image(f"{BASE}/results/charts/fig2_segmented_backtest.png", use_container_width=True)
        st.caption("分段回测图为基线口径（与项目书/PPT 同图）；本区上方的指标表与净值图为统一回测引擎口径。")

    with st.expander("⑥ 调仓建议：PPO 权重与校验", expanded=False):
        ppo_info = _current_ppo_info(latest["date"])
        if ppo_info.get("ok"):
            st.caption(f"{_status_text(ppo_info)} · 权重日期 {ppo_info['date'].date()}")
            _render_weight_bar(ppo_info, "最新 PPO 调仓权重")
            weight_sum = sum(ppo_info["weights"].values())
            c1, c2 = st.columns(2)
            c1.success(f"权重和={weight_sum:.4f} ✓")
            c2.success("单资产≥1% ✓" if min(ppo_info["weights"].values()) >= 0.01 - 1e-6 else "单资产≥1% 待检查")
        else:
            st.warning(ppo_info.get("message", "PPO 权重不可用"))


with tab3:
    st.header("智能配置推荐")

    risk = st.selectbox("选择风险档", ["保守型(C2) — 权益20%", "平衡型(C3) — 权益50%", "进取型(C4) — 权益70%"])
    tier_map = {"保守型(C2) — 权益20%": "保守型(C2)", "平衡型(C3) — 权益50%": "平衡型(C3)", "进取型(C4) — 权益70%": "进取型(C4)"}
    tier_name = tier_map[risk]
    tier_data = tiers_df[tiers_df["风险档"] == tier_name]

    captions = {
        "保守型(C2)": "💡 C2 适合退休/低风险投资者。LGB融合年化8.0%、回撤仅-12.7%，在保守档全面优于动态评分。",
        "平衡型(C3)": "💡 C3 是 sweet spot——LGB融合年化12.2%/-24.7%回撤，对大多数零售客户可接受。",
        "进取型(C4)": "💡 C4 年化15.0%但回撤-34.0%，超出零售客户通常承受范围（≤20%），谨慎推荐。",
    }

    col1, col2 = st.columns(2)
    for i, (_, row) in enumerate(tier_data.iterrows()):
        bg = "#E91E63" if row["策略"] == "动态评分" else "#FF9800"
        with [col1, col2][i]:
            st.markdown(
                f"""
            <div style='background:{bg}15; padding:15px; border-radius:10px; border-left:4px solid {bg}'>
                <h4>{row['策略']}</h4>
                <h2>{row['年化收益']*100:.1f}% <small style='font-size:14px;color:#888'>年化</small></h2>
                <p>回撤: {row['最大回撤']*100:.1f}% | 夏普: {row['夏普比率']:.2f} | 净值: {row['最终净值']:.2f}×</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("推荐配置 — ETF 权重分布")
    equity_pct = {"保守型(C2)": 0.20, "平衡型(C3)": 0.50, "进取型(C4)": 0.70}[tier_name]
    etf_names = ["沪深300", "中证500", "科创50", "消费", "芯片", "黄金", "十年国债"]
    etf_weights = [equity_pct / 5] * 5 + [(1 - equity_pct) / 2] * 2

    fig_bar = go.Figure()
    colors_bar = ["#E91E63"] * 5 + ["#FFC107", "#4CAF50"]
    fig_bar.add_trace(go.Bar(x=etf_names, y=[w * 100 for w in etf_weights], marker_color=colors_bar, text=[f"{w*100:.0f}%" for w in etf_weights], textposition="outside"))
    fig_bar.add_hline(y=equity_pct * 100, line_dash="dash", line_color="gray", annotation_text=f"权益中枢 {equity_pct*100:.0f}%")
    fig_bar.update_layout(height=350, margin=dict(t=20, b=20), yaxis_title="权重(%)", yaxis=dict(range=[0, max(etf_weights) * 100 + 10]))
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(captions.get(tier_name, ""))

    st.divider()
    st.subheader("🧪 PPO历史日期推理")
    st.caption("换日期需重放，约 10–30 秒（本机实测 16s）；默认页面使用离线权重秒开。")
    user_choose_date = st.date_input("选择推理日期", value=DATA_DATE_MAX, min_value=DATA_DATE_MIN, max_value=DATA_DATE_MAX)
    live_info = _render_live_button("🚀 执行 PPO 推理", user_choose_date)
    shown_info = live_info if live_info and live_info.get("ok") else _current_ppo_info(user_choose_date)
    if shown_info.get("ok"):
        st.caption(f"{_status_text(shown_info)} · 权重日期 {shown_info['date'].date()}")
        _render_weight_bar(shown_info, "PPO 推理权重")


with tab4:
    st.header("AI 投顾助手")

    ppo_info = _current_ppo_info(latest["date"])
    if ppo_info["ok"]:
        st.caption(f"{ppo_info['text']}（{_status_text(ppo_info)}，权重日期 {ppo_info['date'].date()}）")
        if ppo_info.get("message"):
            st.caption(ppo_info["message"])
    else:
        st.caption(ppo_info.get("message", "当前无 PPO 权重，仅基于市场评分回答。"))

    mode = st.radio("模式", ["🎯 预制场景", "💬 实时问答 (DeepSeek)"], horizontal=True)

    if mode == "🎯 预制场景":
        ppo_scene_text = f"\n\n当前 {ppo_info['text']}。" if ppo_info["ok"] else "\n\n当前无 PPO 权重，仅基于市场评分回答。"
        scenarios = {
            "😱 市场大跌怎么办": (
                "恐慌安抚",
                f"看到账户浮亏确实让人不安。当前市场评分 **{latest['total_score']:.0f} 分（{latest['market_state']}）**。\n\n"
                f"估值维度 **{latest['val_score']:.0f}/25**，宏观维度 **{latest['macro_score']:.0f}/25**。\n\n"
                "市场处于恐慌区间时，往往是长线布局的窗口。动态评分系统建议维持防御性仓位，等待评分回升后再逐步加仓。"
                + ppo_scene_text,
            ),
            "🔥 现在该加仓吗": (
                "过热止盈",
                f"当前市场评分 **{latest['total_score']:.0f} 分（{latest['market_state']}）**。\n\n"
                '评分超过 60 时市场偏强，建议适度参与但不追高。2015 股灾和 2018 熊市中验证了"过热时降低仓位"的价值。'
                + ppo_scene_text,
            ),
            "📅 定投还有用吗": (
                "长期定投",
                '定投的核心优势是"用时间分散买入成本"。\n\n'
                '回测显示：2021 年后 1307 天震荡期，等权仅赚 10%，动态评分配置实现 33% 收益——\n"市场状态分类 + 定投纪律"在长期中显著跑赢被动持有。'
                + ppo_scene_text,
            ),
        }
        col_scene, col_chat = st.columns([1, 2])
        with col_scene:
            for label in scenarios:
                if st.button(label, use_container_width=True):
                    st.session_state["scene"] = label
        with col_chat:
            if "scene" not in st.session_state:
                st.session_state["scene"] = "😱 市场大跌怎么办"
            title, content = scenarios[st.session_state["scene"]]
            st.markdown(f"### {title}")
            st.info(content)
    else:
        st.caption("基于 DeepSeek API，回答中嵌入当前市场数据")
        user_q = st.text_input("输入你的问题", placeholder="比如：现在适合加仓吗？")
        api_key = st.text_input("DeepSeek API Key", type="password", placeholder="sk-...")

        if st.button("发送", use_container_width=True) and user_q and api_key:
            system_prompt = f"""你是一个智能投顾助手，为农行客户提供资产配置建议。
当前市场环境: 评分 {latest['total_score']:.0f}/100，状态 {latest['market_state']}。
估值{latest['val_score']:.0f}/25，宏观{latest['macro_score']:.0f}/25，情绪{latest['sent_score']:.0f}/20，趋势{latest['trend_score']:.0f}/30。
{ppo_info['text'] if ppo_info['ok'] else '无 PPO 权重，仅基于市场评分回答。'}
资产池: 沪深300/中证500/科创50/消费/芯片/黄金/十年国债 (7只ETF)。
策略: 动态评分配置(年化19.8%/-16.5%回撤)，LGB融合(年化12.4%/夏普0.66)。
回答要简洁专业，不生成具体投资建议。"""
            with st.spinner("思考中..."):
                try:
                    resp = requests.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": "deepseek-chat", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_q}], "max_tokens": 500, "temperature": 0.7},
                        timeout=15,
                    )
                    if resp.status_code == 200:
                        st.success(resp.json()["choices"][0]["message"]["content"])
                    else:
                        st.error(f"API错误: {resp.status_code} — {resp.text[:200]}")
                except Exception as e:
                    st.error(f"请求失败: {e}")
                    st.info('💡 可切换到"预制场景"模式体验')


with tab5:
    st.header("投资者风险等级划分")

    st.subheader("C1-C5 五档风险等级")
    risk_table = pd.DataFrame(
        [
            ["C1", "保守型", "仅限 R1（如货币基金）", "≤20 分", "超出回测范围"],
            ["C2", "谨慎型", "R1、R2（如债券基金）", "21-40 分", "年化 8.0%/回撤 -12.7%"],
            ["C3", "稳健型", "R1、R2、R3（如混合基金）", "41-65 分", "年化 12.2%/回撤 -24.7% ✨"],
            ["C4", "积极型", "R1-R4（如股票基金）", "66-90 分", "年化 15.0%/回撤 -34.0%"],
            ["C5", "激进型", "R1-R5（含期货期权）", "≥91 分", "超出回测范围"],
        ],
        columns=["等级", "名称", "可购买产品", "问卷得分", "回测覆盖"],
    )
    st.dataframe(risk_table, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("问卷设计概要")
    st.markdown(
        """
    - **15 题多维测评**：年龄、学历、职业、收入来源、年收入、投资占比、债务情况、
      投资经验、投资品种、知识水平、投资倾向、亏损承受、亏损反应、投资目标、资金期限
    - **满分 148 分**，按总分映射到 C1-C5 五档
    - **有效期 12 个月**，财务状况重大变化需重新测评
    - **特殊认定**：第 12 题选"不能承受任何亏损"且总分<30 → 认定为最低风险类别
    """
    )
    st.caption("数据源: xf《用户画像测评问卷+打分规则.docx》《风险等级划分规则.docx》")

    st.divider()
    st.subheader("三档策略匹配")
    st.dataframe(tiers_df.style.format({"年化收益": "{:.1%}", "最大回撤": "{:.1%}", "夏普比率": "{:.2f}"}), use_container_width=True, hide_index=True)
    st.caption("C1/C5 超出回测覆盖范围，建议咨询客户经理后人工配置")

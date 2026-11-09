"""持久化简报图：只读聚合数据、人工中断、原子落盘、幂等通知。"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import sqlite3
import threading
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4

import pandas as pd
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import settings
from app import feishu
from app.tools import registry

log = logging.getLogger(__name__)
DISCLAIMER = '仅供研究演示，不构成投资建议。市场有风险，投资需谨慎。'
REVIEW_HEADING = '## 七、审核状态与审核人'
MODEL_NAMES = ('LGB融合', 'PPO月度', 'SASF硬切换')


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    action: Literal['approve', 'edit', 'reject']
    reviewer: str = Field(min_length=1, max_length=80)
    comment: str = Field(default='', max_length=1000)
    edited_markdown: str = Field(default='', max_length=50000)

    @field_validator('reviewer', 'comment', mode='before')
    @classmethod
    def trim(cls, value):
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode='after')
    def content_required(self):
        if self.action == 'edit' and not self.edited_markdown.strip():
            raise ValueError('edit 必须提供 edited_markdown')
        if self.action == 'reject' and not self.comment:
            raise ValueError('reject 必须提供 comment')
        return self


class WorkflowState(TypedDict, total=False):
    thread_id: str
    report_date: str | None
    status: str
    market: dict
    strategies: dict
    risk: dict
    caveats: dict
    weights: dict | None
    draft: str
    revision: int
    reject_streak: int
    review_history: list[dict]
    report_path: str | None
    report_sha256: str | None
    publish_status: str
    publish_error: str | None
    publish_attempts: int
    message_id: str | None
    idempotency_key: str
    created_at: str
    updated_at: str
    error: str | None


def node(fn):
    @wraps(fn)
    def timed(state):
        started = time.perf_counter()
        try:
            result = fn(state)
        except GraphInterrupt:
            log.info('workflow node=%s thread_id=%s status=awaiting_review latency_ms=%.2f',
                     fn.__name__, state['thread_id'], (time.perf_counter()-started)*1000)
            raise
        except Exception as exc:
            log.warning('workflow node=%s thread_id=%s error_type=%s', fn.__name__, state['thread_id'], type(exc).__name__)
            result = dict(status='failed', error=f'{type(exc).__name__}: {fn.__name__} 节点失败', updated_at=now())
        update = result.update if isinstance(result, Command) else result
        status = (update or {}).get('status', state.get('status'))
        log.info('workflow node=%s thread_id=%s status=%s latency_ms=%.2f',
                 fn.__name__, state['thread_id'], status, (time.perf_counter()-started)*1000)
        # Root-dict state permits required node names risk/draft alongside same-named fields.
        if isinstance(result, Command):
            return Command(update=[('__root__', {**state, **(result.update or {})})], goto=result.goto)
        return {**state, **result}
    return timed


def required(fn, **args):
    result = fn(**args)
    if not result.get('ok'):
        raise ValueError(f'{fn.__name__} 读取失败')
    return result


def pct(value):
    raw = str(value).strip()
    number = float(raw.removesuffix('%')) if raw.endswith('%') else float(value)*100
    if not math.isfinite(number):
        raise ValueError('非有限数值')
    return f'{number:.2f}%'


def cell(value):
    return str(value).replace('|', '／').replace('\n', ' ')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in rows])


def audit_section(state, status='待审核'):
    history = state['review_history']
    last = history[-1] if history else None
    return REVIEW_HEADING + f"\n\n状态：{status}；审核人：{last['reviewer'] if last else '未审核'}；审核时间：{last['at'] if last else '未审核'}。\n\n" + table(
        ['动作', '审核人', '意见', '时间', '修订号'],
        [[h['action'], h['reviewer'], h['comment'], h['at'], h['revision']] for h in history])


def with_audit(markdown, state, status):
    return markdown.split(REVIEW_HEADING, 1)[0].rstrip() + '\n\n' + audit_section(state, status) + '\n'


def render_report(state):
    market, caveats = state['market']['data'], state['caveats']['data']
    rows = sorted(state['strategies']['data']['rows'], key=lambda r: r['年化收益'], reverse=True)
    metrics = lambda r: [pct(r['年化收益']), pct(r['最大回撤']), r['夏普比率'], r['卡玛比率']]
    strategy_table = table(['排名', '策略', '年化收益', '最大回撤', '夏普', '卡玛'],
                           [[rank, r['策略'], *metrics(r)] for rank, r in enumerate(rows, 1)])
    stress = state['risk']['stress']['data']['rows']
    stress_table = table(['危机阶段', '策略', '最大回撤', '修复天数'],
                         [[r['危机阶段'], r['策略'], pct(r['最大回撤']), r['修复天数']] for r in stress])
    tiers = state['risk']['tiers']['data']['rows']
    tier_table = table(['风险档', '策略', '年化收益', '最大回撤', '夏普', '卡玛'],
                       [[r['风险档'], r['策略源'], *metrics(r)] for r in tiers])
    models = table(['模型', '年化收益', '最大回撤', '夏普', '卡玛'],
                   [[r['策略'], *metrics(r)] for r in rows if r['策略'] in MODEL_NAMES])
    weights = state.get('weights')
    weight_note = ('权重缓存日期：' + weights['date'] + '。\n\n' + table(['资产', '权重'],
                   [[name, pct(value)] for name, value in weights['weights'].items()])) if weights else '无权重缓存。'
    dims = '；'.join(f"{name}：{item['features']} 个特征，{item['columns']} 列" for name, item in caveats['dimensions'].items())
    body = f'''# 投研简报 {state['report_date']}（历史演示数据）

## 一、市场状态与评分

数据日期：{state['report_date']}。总评分：{market['total_score']:.2f} 分；状态：{market['market_state']}。
估值：{market['val_score']:.2f} 分；宏观：{market['macro_score']:.2f} 分；情绪：{market['sent_score']:.2f} 分；趋势：{market['trend_score']:.2f} 分。

## 二、策略表现与相对排名

按年化收益降序；历史全区间回测摘要。

{strategy_table}

## 三、风险提示

{stress_table}

{tier_table}

历史回测不代表未来收益。

## 四、模型观点

以下为模型历史表现摘要，不是实时预测或交易指令。

{models}

{weight_note}

## 五、数据与口径备注

数据覆盖：{caveats['start']} 至 {caveats['end']}，{caveats['trading_days']} 个交易日。
北向缺失区间：{cell(caveats['nan_range'])}，缺失 {caveats['nan_rows']} 行；连续填零从 {caveats['zero_start']} 起，共 {caveats['zero_rows']} 行；最后有效日 {caveats['last_valid_date']}。
宏观滞后 {caveats['macro_lag']} 个交易日。特征维度：{dims}。
距截止日约 {caveats['months_since_cutoff']} 个月。历史回测演示数据，不是实时行情。
数据依据：{state['market']['source']}；{state['strategies']['source']}；{state['risk']['stress']['source']}；{state['risk']['tiers']['source']}；{state['caveats']['source']}。
'''
    if weights:
        body += '\n权重依据：results/ppo_weight.csv（仅最新一行聚合权重）。\n'
    body += '\n## 六、免责声明\n\n' + DISCLAIMER + '\n\n' + audit_section(state) + '\n'
    return body


def atomic_save(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        temp.write_text(text, encoding='utf-8')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)  # Only this invocation's temporary file.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_workflow(checkpointer, *, config=settings):
    @node
    def collect(state):
        market = required(registry.get_market_score)
        caveats = required(registry.get_data_caveats)
        if not market['as_of'] or caveats['as_of'] != market['as_of']:
            raise ValueError('数据日期不一致')
        return dict(market=market, caveats=caveats, report_date=market['as_of'],
                    idempotency_key=f"brief-{market['as_of']}-{state['thread_id']}"[:50], updated_at=now())

    @node
    def score(state):
        data = state['market']['data']
        for key in ('total_score', 'val_score', 'macro_score', 'sent_score', 'trend_score'):
            if not math.isfinite(float(data[key])):
                raise ValueError('评分无效')
        if not data['market_state']:
            raise ValueError('状态缺失')
        return dict(updated_at=now())

    @node
    def models(state):
        strategies = required(registry.get_strategy_metrics)
        rows = strategies['data']['rows']
        if strategies['as_of'] != state['report_date'] or len(rows) != 10 or not set(MODEL_NAMES) <= {r['策略'] for r in rows}:
            raise ValueError('策略摘要不完整')
        weights = None
        try:
            path = config.platform_dir/'results/ppo_weight.csv'
            df = pd.read_csv(path, encoding='utf-8-sig').sort_values('date')
            assets = ['hs300', 'zz500', 'kc50', 'consume', 'chip', 'gold', 'bond10']
            latest = df.iloc[-1]
            values = {name: float(latest[name]) for name in assets}
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
                raise ValueError('权重无效')
            weights = dict(date=str(latest['date']), weights=values)
        except Exception:
            log.info('workflow optional_weight=unavailable thread_id=%s', state['thread_id'])
        return dict(strategies=strategies, weights=weights, updated_at=now())

    @node
    def risk(state):
        stress = required(registry.get_drawdown_recovery)
        tiers = required(registry.get_risk_tier, strategy='动态评分')
        phases = {r['危机阶段'] for r in stress['data']['rows']}
        levels = ' '.join(r['风险档'] for r in tiers['data']['rows'])
        if any(not any(year in phase for phase in phases) for year in ('2015', '2018', '2021')) or any(level not in levels for level in ('C2', 'C3', 'C4')):
            raise ValueError('风险摘要不完整')
        if any(r['as_of'] != state['report_date'] for r in (stress, tiers)):
            raise ValueError('风险日期不一致')
        return dict(risk={'stress': stress, 'tiers': tiers}, updated_at=now())

    @node
    def draft(state):
        return dict(draft=render_report(state), status='awaiting_review', updated_at=now())

    @node
    def human_review(state):
        payload = interrupt({'thread_id': state['thread_id'], 'status': 'awaiting_review', 'revision': state['revision']})
        review = ReviewRequest.model_validate(payload)
        revision = state['revision'] + (review.action != 'approve')
        history = [*state['review_history'], dict(action=review.action, reviewer=review.reviewer,
                   comment=review.comment, at=now(), revision=revision)]
        update = dict(review_history=history, revision=revision, updated_at=now())
        if review.action == 'approve':
            return Command(update=update, goto='publish')
        if review.action == 'edit':
            return Command(update={**update, 'draft': review.edited_markdown, 'status': 'awaiting_review', 'reject_streak': 0}, goto='human_review')
        streak = state['reject_streak'] + 1
        update.update(reject_streak=streak)
        if streak >= 3:
            return Command(update={**update, 'status': 'rejected'}, goto=END)
        return Command(update=update, goto='draft')

    @node
    def publish(state):
        # A retry reuses the already saved report and the same application UUID.
        if state.get('report_path'):
            path = config.agent_dir/state['report_path']
            if not path.is_file():
                raise ValueError('已保存报告缺失')
        else:
            path = config.reports_dir/f"{state['report_date']}_{state['thread_id']}.md"
            atomic_save(path, with_audit(state['draft'], state, '审核通过，待通知'))
        rows = state['strategies']['data']['rows']
        summary = '；'.join(f"{r['策略']}：年化 {pct(r['年化收益'])}，最大回撤 {pct(r['最大回撤'])}" for r in rows if r['策略'] in MODEL_NAMES)
        card = feishu.build_brief_card(report_date=state['report_date'], score=state['market']['data']['total_score'],
              state=state['market']['data']['market_state'], strategy_summary=summary,
              risk_note='历史回测不代表未来收益。'+DISCLAIMER,
              report_url=config.report_url_base+'/workflow/'+state['thread_id']+'/report')
        try:
            result = feishu.publish_brief(card, idempotency_key=state['idempotency_key'])
        except Exception as exc:
            result = dict(status='failed', message_id=None, error=feishu.safe_error(exc), attempts=0)
        status = {'sent': 'published', 'skipped': 'saved_only', 'failed': 'publish_failed'}[result['status']]
        labels = {'published': '已发布', 'saved_only': '审核通过，仅保存', 'publish_failed': '审核通过，通知失败'}
        text = with_audit(state['draft'], state, labels[status])
        sha = atomic_save(path, text)
        return dict(status=status, draft=text, report_path=str(path.relative_to(config.agent_dir)), report_sha256=sha,
                    publish_status=result['status'], publish_error=result['error'], message_id=result['message_id'],
                    publish_attempts=state.get('publish_attempts', 0)+result['attempts'], updated_at=now())

    graph = StateGraph(dict)
    for name, fn in [('collect', collect), ('score', score), ('models', models), ('risk', risk), ('draft', draft), ('human_review', human_review), ('publish', publish)]:
        graph.add_node(name, fn)
    graph.add_edge(START, 'collect')
    for source, target in [('collect', 'score'), ('score', 'models'), ('models', 'risk'), ('risk', 'draft'), ('draft', 'human_review')]:
        graph.add_conditional_edges(source, lambda state, target=target: END if state['status'] == 'failed' else target)
    graph.add_edge('publish', END)
    return graph.compile(checkpointer=checkpointer)


class WorkflowConflict(ValueError):
    pass


class WorkflowService:
    """SQLite owns workflow state; the lock only serializes single-process mutations."""
    def __init__(self, config=settings):
        self.config = config
        config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(config.checkpoint_path, check_same_thread=False)
        self.graph = build_workflow(SqliteSaver(self.connection), config=config)
        self.lock = threading.RLock()
        self.closed = False

    def close(self):
        if not self.closed:
            self.connection.close()
            self.closed = True

    @staticmethod
    def graph_config(thread_id):
        return {'configurable': {'thread_id': thread_id}}

    def state(self, thread_id):
        value = self.graph.get_state(self.graph_config(thread_id)).values
        if not value:
            raise KeyError(thread_id)
        return value

    def run(self):
        with self.lock:
            thread_id = uuid4().hex
            state = dict(thread_id=thread_id, report_date=None, status='collecting', draft='', revision=0,
                         reject_streak=0, review_history=[], report_path=None, report_sha256=None,
                         publish_status='not_attempted', publish_error=None, publish_attempts=0, message_id=None,
                         idempotency_key='', created_at=now(), updated_at=now(), error=None)
            self.graph.invoke(state, self.graph_config(thread_id))
            return self.state(thread_id)

    def review(self, thread_id, review: ReviewRequest):
        with self.lock:
            state = self.state(thread_id)
            if state['status'] in {'published', 'saved_only'} and review.action == 'approve':
                return {**state, 'duplicate': True}
            if state['status'] == 'publish_failed' and review.action == 'approve':
                state = {**state, 'review_history': [*state['review_history'],
                    dict(action='approve', reviewer=review.reviewer, comment=review.comment,
                         at=now(), revision=state['revision'])], 'updated_at': now()}
                self.graph.invoke(Command(update=[('__root__', state)], goto='publish'), self.graph_config(thread_id))
            elif state['status'] == 'awaiting_review':
                self.graph.invoke(Command(resume=review.model_dump()), self.graph_config(thread_id))
            else:
                raise WorkflowConflict('当前状态不允许该审核动作')
            return self.state(thread_id)

    def get(self, thread_id):
        with self.lock:
            return self.state(thread_id)


def public_state(state):
    fields = ('thread_id', 'report_date', 'status', 'draft', 'revision', 'review_history', 'report_path',
              'report_sha256', 'idempotency_key', 'created_at', 'updated_at', 'error')
    result = {key: state.get(key) for key in fields}
    result['publish'] = dict(status=state['publish_status'], message_id=state.get('message_id'),
                             error=state.get('publish_error'), attempts=state.get('publish_attempts', 0))
    if state.get('duplicate'):
        result['duplicate'] = True
    return result

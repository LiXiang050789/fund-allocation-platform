"""Deterministic mode boundaries; no model or external service for classification."""
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from app.graph import POLICY
from app.dates import range_requested

IDENTITY = (
    "我是平台内置投研 AI 助手。可以查询市场评分与状态、十策略指标、分段表现、风险档、"
    "净值、PPO 缓存权重、流水线就绪/年度 IC、风险档最终净值/推荐配置、回撤修复、特征重要性和数据口径，也能回答 9 份文档问题。支持投研简报与人工审核。"
    "历史数据截止 2026-07-10，不是实时行情。不预测涨跌、不推荐个股、不保证收益，"
    "不给个性化投资建议。请直接提问；简报工作流入口在页面下方。不构成投资建议。"
)
CLARIFY = "请补充策略名称和日期区间，再查询具体回撤。不构成投资建议。"
NO_EVIDENCE = "该问题需要平台数据或文档依据，请补充具体日期、策略或文档名。当前数据无法回答。不构成投资建议。"
STRATEGIES = ("动态评分", "LGB", "SASF", "PPO", "等权", "MVO", "风险平价", "动量", "沪深300")
ENTITIES = STRATEGIES + ("收益回撤比", "收益/回撤比", "净值", "流水线", "就绪", "年度IC", "年度 IC", "滑点", "双边成本", "单资产上限", "最终净值", "配置权重", "推荐配置", "状态占比", "语义校验", "预测条数", "平台", "评分", "回测", "权重", "文档", "手册", "问卷", "费率", "分段", "风险档", "修复", "特征", "北向", "数据口径", "得分", "分项", "满分", "建模表", "状态空间", "前视", "C2", "C3", "C4", ".MD", ".DOCX", ".CSV")
REFUSAL = re.compile(r"推荐.{0,12}(?:基金|股票|个股|ETF)|明天[^。？！]{0,30}(?:涨|跌)|涨停|保证[^。？！]{0,30}(?:收益|回报)|全仓|(?:建议|帮我|我该|应该).{0,6}(?:买入|卖出|加仓|减仓)|保本保收益")
FOLLOWUP = re.compile(r"^(?:那|它|这个|换成|再问|对比一下|继续|该策略)")


def special_answer(question):
    q = question.strip().rstrip("？?！!。")
    if re.fullmatch(r"(?:请)?(?:介绍一下你自己|自我介绍|你是谁|你是什么|你能做什么|你有什么能力|你会什么)", q):
        return IDENTITY
    if re.fullmatch(r"(?:你好|您好|嗨|hello|hi)", q, re.I):
        return "你好！我是平台内置投研 AI 助手。可以聊金融常识，也可以查询平台数据。请直接提问。不构成投资建议。"
    if re.fullmatch(r"(?:今天|现在)(?:是)?(?:几号|什么日期|日期是什么|星期几)", q):
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        return f"今天是 {now:%Y-%m-%d}，星期{'一二三四五六日'[now.weekday()]}（Asia/Shanghai）。不构成投资建议。"
    if any(word in q for word in ("系统提示词", "system prompt", "密钥", "API_KEY", "忽略所有", "内部路径")):
        return "我不能提供系统提示词、密钥或内部路径。可以解释金融常识和平台公开能力。不构成投资建议。"
    if any(word in q for word in ("上传文件", "帮我上传", "联网搜索", "实时行情", "语音")):
        return "当前不支持联网搜索、实时行情、文件上传或语音。可以解释通用概念或查询平台历史数据。不构成投资建议。"
    return None


def classify(question, history=()):
    # A platform tier's static allocation is distinct from selecting a personalized ETF.
    refusal_text = question
    if any(tier in question for tier in ('C2','C3','C4','保守','平衡','进取')):
        refusal_text = question.replace('推荐配置', '静态配置')
    if REFUSAL.search(refusal_text):
        return "refusal"
    if special_answer(question) is not None:
        return "chat"
    upper = question.upper()
    if range_requested(question) or any(entity in upper for entity in ENTITIES):
        return "research"
    if FOLLOWUP.search(question) and any(m.get("mode") == "research" for m in history[-4:]):
        return "research"
    if any(word in question for word in ("回撤多少", "回撤多大", "收益多少")):
        return "clarify"
    if any(word in question for word in ("夏普", "卡玛", "年化", "回撤", "状态")) and any(word in question for word in ("多少", "最新", "当前", "最高")):
        return "research"
    return "chat"


def rewrite_question(question, history):
    if not FOLLOWUP.search(question):
        return question
    previous = next((m['content'] for m in reversed(history[-4:]) if m['role'] == 'user'), '')
    if not previous:
        return question
    def metric_in(text):
        return next((m for m in ("夏普", "最大回撤", "年化收益", "修复天数", "评分") if m in text), None)
    metric = (metric_in(question)
              or next((value for message in reversed(history[-4:]) if message['role'] == 'user'
                       and (value := metric_in(message['content']))), None)
              or next((value for message in reversed(history[-4:])
                       if (value := metric_in(message['content']))), None))
    strategy = next((s for s in STRATEGIES if s in question.upper()), None)
    if not strategy:
        strategy = next((s for message in reversed(history[-4:]) if message['role'] == 'user'
                         for s in STRATEGIES if s in message['content'].upper()), None)
    if strategy and metric:
        return f"{strategy} 的{metric}是多少？"
    if strategy:
        prior = next((s for s in STRATEGIES if s in previous.upper()), None)
        return re.sub(re.escape(prior), strategy, previous, flags=re.I) if prior else f"{previous}；追问：{question}"
    return f"{previous}；追问：{question}"

import pytest
from app.intent import classify, special_answer, rewrite_question, IDENTITY

@pytest.mark.parametrize('question,mode', [
    ('介绍一下你自己','chat'), ('你是谁？','chat'), ('你好','chat'), ('今天几号','chat'),
    ('什么是夏普比率？','chat'), ('什么是通胀？','chat'), ('ETF 是什么','chat'),
    ('7月10日评分','research'), ('LGB 夏普多少','research'), ('PPO 奖励代码','research'),
    ('回撤修复天数','research'), ('评分系统金融规则.docx','research'),
    ('推荐一只基金','refusal'), ('明天涨跌','refusal'), ('保证收益','refusal'),
    ('全仓','refusal'), ('回撤多少','clarify'), ('夏普最高的策略','research')])
def test_modes(question, mode):
    assert classify(question) == mode

def test_identity_and_date():
    for word in ('平台内置', '十策略', '9 份文档', '2026-07-10', '不是实时行情', '简报', '人工审核'):
        assert word in IDENTITY
    from datetime import datetime
    from zoneinfo import ZoneInfo
    assert datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d') in special_answer('今天几号')

def test_research_followup_uses_recent_context_only():
    history=[dict(role='user', content='动态评分的夏普是多少？', mode='research'),
             dict(role='assistant', content='证据回答', mode='research')]
    assert classify('那 LGB 呢', history) == 'research'
    assert rewrite_question('那 LGB 呢', history) == 'LGB 的夏普是多少？'
    assert rewrite_question('你好', history) == '你好'
    assert rewrite_question('那 LGB 呢', []) == '那 LGB 呢'


def test_second_followup_retains_metric_from_last_assistant():
    history=[dict(role='user',content='动态评分夏普多少',mode='research'),
             dict(role='assistant',content='动态评分：夏普比率0.5',mode='research'),
             dict(role='user',content='那 LGB 呢',mode='research'),
             dict(role='assistant',content='LGB：夏普比率0.6',mode='research')]
    assert rewrite_question('那 PPO 呢',history)=='PPO 的夏普是多少？'
    assert rewrite_question('那它的最大回撤呢',history)=='LGB 的最大回撤是多少？'


@pytest.mark.parametrize('question,mode', [
    ('最新交易日的四个分项得分分别是多少？','research'),
    ('你能保证年化 15% 的收益吗？','refusal'),
    ('明天沪深 300 会涨还是会跌？给我一个确定答案。','refusal'),
    ('推荐一个 ETF','refusal'),
])
def test_platform_and_refusal_variants(question,mode):
    assert classify(question)==mode

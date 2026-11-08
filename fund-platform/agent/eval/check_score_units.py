"""真实 HTTP 评分单位复测；服务由调用方启动，逐次运行。"""
import argparse
import json
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.graph import score_unit_error
from app.tools.registry import get_market_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:8001')
    args = parser.parse_args()
    expected = get_market_score()
    assert expected['ok'], expected['error']
    score = expected['data']['total_score']
    questions = ['最新市场评分和状态'] * 5 + ['最新一个交易日的市场评分是多少分？'] * 3
    failed = 0
    for i, question in enumerate(questions, 1):
        response = requests.post(args.url + '/chat', json={'question': question}, timeout=180)
        response.raise_for_status()
        data = response.json()
        answer = data['answer']
        actual = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', answer)]
        passed = data['verified'] and score in actual and not score_unit_error(answer, [expected])
        failed += not passed
        print(json.dumps(dict(index=i, question=question, passed=passed, response=data), ensure_ascii=False), flush=True)
    print(f'score_unit_checks={len(questions)-failed}/{len(questions)}', flush=True)
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())

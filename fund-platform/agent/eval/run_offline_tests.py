"""Run pytest with ALL outbound sockets blocked and mocked-publication tests enabled.

This changes only the in-memory test Settings. Production and .env stay unchanged.
Existing workflow fixtures replace the sender; they expect the normal publication
branch. Setting AGENT_FEISHU_DISABLED=1 globally changes those expected branches.
"""
import os
import socket
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['AGENT_FEISHU_DISABLED'] = '1'


def blocked(*args, **kwargs):
    raise RuntimeError('offline tests: outbound sockets blocked')


if __name__ == '__main__':
    socket.socket.connect = blocked
    import app.config
    app.config.settings = replace(app.config.settings, feishu_disabled=False)
    import pytest
    raise SystemExit(pytest.main(sys.argv[1:] or ['-q']))

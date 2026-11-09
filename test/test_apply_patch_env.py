"""apply_patch 环境组合判别测试（B-5 整改钉子）。

B-5：组合 ENABLE_UCM_PATCH=0 + ENABLE_GSA_PATCH=1 原先静默无补丁、无任何日志。
修复后 GSA_PATCH 被强制关闭（依赖 UCM 底座）并打 WARNING 留痕。

本机（无 NPU 构建）缺少 C++ 扩展 ucm.shared.infra.ucmlogger，按
test/test_logger.py 的惯例预置 stub 后再导入真实包链（真实环境存在该模块时
保留原样）；ucm 根 logger propagate=False，故手挂 Handler 采集日志而非 caplog。
本机运行：py -3 -X utf8 -m pytest --noconftest -q test\\test_apply_patch_env.py
"""

import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# C++ 扩展 stub：必须先于任何 ucm 导入预置。
_backend = ModuleType("ucm.shared.infra.ucmlogger")
_backend.Level = SimpleNamespace(DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50)
_backend.setup = lambda *a: None
_backend.flush = lambda: None
_backend.log = lambda *a: None
_backend.log_rate_limit = lambda *a: None
sys.modules.setdefault("ucm.shared.infra.ucmlogger", _backend)

import ucm.integration.vllm.patch.apply_patch as apply_patch  # noqa: E402


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _reload_with_env(monkeypatch, ucm_flag, gsa_flag):
    monkeypatch.setenv("ENABLE_UCM_PATCH", ucm_flag)
    monkeypatch.setenv("ENABLE_GSA_PATCH", gsa_flag)
    return importlib.reload(apply_patch)


def _run_and_capture(mod):
    """执行 apply_all_patches 并采集 mod.logger 日志（propagate=False）。"""
    handler = _ListHandler()
    mod.logger.addHandler(handler)
    try:
        mod.apply_all_patches()
    finally:
        mod.logger.removeHandler(handler)
    return handler.records


class TestPatchEnvCombination:
    def test_ucm_off_gsa_on_forces_gsa_off_with_warning(self, monkeypatch):
        mod = _reload_with_env(monkeypatch, "0", "1")
        # B-5：原始请求被记录，但有效旗标被 UCM 底座强制关闭。
        assert mod.GSA_PATCH_REQUESTED is True
        assert mod.ENABLE_UCM_PATCH is False
        assert mod.ENABLE_GSA_PATCH is False
        records = _run_and_capture(mod)  # 门控早退，不触碰任何 vllm 导入
        warnings = [r for r in records if r.levelno == logging.WARNING]
        assert any("ignored" in r.getMessage() for r in warnings)

    def test_ucm_off_gsa_off_logs_info_not_silent(self, monkeypatch):
        mod = _reload_with_env(monkeypatch, "0", "0")
        assert mod.ENABLE_GSA_PATCH is False
        records = _run_and_capture(mod)
        assert any("no vLLM/vllm-ascend patches" in r.getMessage() for r in records)
        assert not [r for r in records if r.levelno >= logging.ERROR]

    def test_ucm_on_gsa_on_keeps_gsa_enabled(self, monkeypatch):
        mod = _reload_with_env(monkeypatch, "1", "1")
        assert mod.ENABLE_UCM_PATCH is True
        assert mod.ENABLE_GSA_PATCH is True
        # 不调用 apply_all_patches：UCM 开启会走真实补丁导入路径（需 vllm 环境）。

    def test_ucm_on_gsa_off_keeps_gsa_disabled(self, monkeypatch):
        mod = _reload_with_env(monkeypatch, "1", "0")
        assert mod.ENABLE_UCM_PATCH is True
        assert mod.GSA_PATCH_REQUESTED is False
        assert mod.ENABLE_GSA_PATCH is False

# -*- coding: utf-8 -*-
"""GSA 共享内存前缀与残留段回收（ucm.sparse.utils）的单元测试。

覆盖：
- get_shm_prefix：engine_id 缺失/清洗/截断；
- _cleanup_stale_gsa_shm：参考 UCM CleanUpShmFileExceptMe 的残留段回收
  判据（其它 engine_id 前缀 + 同 uid + 超过竞态保护窗口 + 无存活持有者）；
- rm_shared_memory：精确名清理带前缀，并触发残留段回收。
"""

import types

from ucm.sparse import utils


class TestGetShmPrefix:
    def test_empty_when_engine_id_missing(self):
        assert utils.get_shm_prefix(None) == ""
        assert utils.get_shm_prefix("") == ""

    def test_prefix_format(self):
        assert utils.get_shm_prefix("svc-a") == "ucm_gsa_svc-a_"

    def test_sanitizes_and_truncates(self):
        assert utils.get_shm_prefix("a/b c") == "ucm_gsa_a_b_c_"
        truncated = "x" * (utils.GSA_SHM_UNIQUE_ID_MAX_LEN + 10)
        expected = "ucm_gsa_" + "x" * utils.GSA_SHM_UNIQUE_ID_MAX_LEN + "_"
        assert utils.get_shm_prefix(truncated) == expected


class TestCleanupStaleGsaShm:
    NOW = 10000.0

    def _run(self, tmp_path, entries, live=(), current="ucm_gsa_cur_", uid=1000):
        """entries: {文件名: mtime}；uid 为文件的属主 uid（euid 固定 1000）。

        返回被删除的文件名列表。
        """
        removed = []

        def fake_listdir(shm_dir):
            assert shm_dir == str(tmp_path)
            return list(entries)

        def fake_stat(path):
            name = path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            return types.SimpleNamespace(st_uid=uid, st_mtime=entries[name])

        def fake_remove(path):
            name = path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            removed.append(name)

        utils._cleanup_stale_gsa_shm(
            current,
            shm_dir=str(tmp_path),
            live_names=set(live),
            now=self.NOW,
            euid=1000,
            _listdir=fake_listdir,
            _stat=fake_stat,
            _remove=fake_remove,
        )
        return removed

    def test_removes_stale_segments_of_other_engine_ids(self, tmp_path):
        for base in utils.GSA_SHM_BASE_NAMES:
            entries = {f"ucm_gsa_old_{base}_dp0": 0.0}
            assert self._run(tmp_path, entries) == [f"ucm_gsa_old_{base}_dp0"]

    def test_removes_legacy_unprefixed_segments_when_prefix_active(self, tmp_path):
        entries = {f"{base}_dp0": 0.0 for base in utils.GSA_SHM_BASE_NAMES}
        removed = self._run(tmp_path, entries)
        assert sorted(removed) == sorted(entries)

    def test_keeps_segments_mapped_by_live_processes(self, tmp_path):
        entries = {"ucm_gsa_old_gsa_glm52_slot_dp0": 0.0}
        live = {"ucm_gsa_old_gsa_glm52_slot_dp0"}
        assert self._run(tmp_path, entries, live=live) == []

    def test_keeps_recently_created_segments(self, tmp_path):
        entries = {"ucm_gsa_old_gsa_glm52_slot_dp0": self.NOW - 10.0}
        assert self._run(tmp_path, entries) == []

    def test_keeps_current_service_segments(self, tmp_path):
        entries = {
            "ucm_gsa_cur_gsa_glm52_slot_dp0": 0.0,
            "ucm_gsa_cur_gsa_glm52_slot_dp7": 0.0,
        }
        assert self._run(tmp_path, entries) == []

    def test_keeps_segments_of_other_users(self, tmp_path):
        entries = {"ucm_gsa_old_gsa_glm52_slot_dp0": 0.0}
        assert self._run(tmp_path, entries, uid=1001) == []

    def test_ignores_non_gsa_names(self, tmp_path):
        entries = {
            "uc_shm_mooncake_abc": 0.0,
            "ucm_gsa_cur_other_dp0": 0.0,
            "ucm_gsa_old_gsa_glm52_slot_dp": 0.0,
            "not_gsa_at_all": 0.0,
        }
        assert self._run(tmp_path, entries) == []

    def test_legacy_current_prefix_keeps_legacy_names(self, tmp_path):
        # 当前服务未配置 engine_id（空前缀）时，历史命名段属于当前服务，
        # 不做模式回收（由精确名清理负责）；其它前缀的残留段照常回收。
        entries = {
            "gsa_glm52_slot_dp0": 0.0,
            "ucm_gsa_x_gsa_glm52_slot_dp0": 0.0,
        }
        assert self._run(tmp_path, entries, current="") == [
            "ucm_gsa_x_gsa_glm52_slot_dp0"
        ]


class TestRmSharedMemory:
    def test_exact_cleanup_uses_prefix_and_triggers_stale_gc(self, monkeypatch):
        seen_names = []
        gc_prefixes = []

        class FakeSharedMemory:
            def __init__(self, name, create=False):
                seen_names.append(name)
                raise FileNotFoundError(name)

        monkeypatch.setattr(utils.shared_memory, "SharedMemory", FakeSharedMemory)
        monkeypatch.setattr(
            utils, "_cleanup_stale_gsa_shm", lambda prefix: gc_prefixes.append(prefix)
        )

        utils.rm_shared_memory(3, "svc-a")

        assert "ucm_gsa_svc-a_gsa_prefetch_result_dp3" in seen_names
        assert len(seen_names) == 6
        assert gc_prefixes == ["ucm_gsa_svc-a_"]

import os
import stat

import pytest

from reachy_motion import userconfig


def test_secrets_are_write_only_and_file_is_private(tmp_path, monkeypatch):
    path = tmp_path / "cfg" / ".env"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    userconfig.write({"OPENAI_API_KEY": "sk-test-123", "ELEVENLABS_AGENT_ID": "agent_abc"}, path)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    view = userconfig.public_view(path)
    assert view["fields"]["OPENAI_API_KEY"] == {**view["fields"]["OPENAI_API_KEY"], "set": True, "value": None}
    assert "sk-test-123" not in repr(view)
    assert view["fields"]["ELEVENLABS_AGENT_ID"]["value"] == "agent_abc"
    assert os.environ["OPENAI_API_KEY"] == "sk-test-123"
    userconfig.write({"OPENAI_API_KEY": None}, path)  # clear
    assert "OPENAI_API_KEY" not in userconfig.read(path)
    assert "OPENAI_API_KEY" not in os.environ


@pytest.mark.parametrize("updates", [{"PATH": "/evil"}, {"OPENAI_API_KEY": "a\nPATH=/evil"}, {"ELEVENLABS_AGENT_ID": "x" * 500}])
def test_rejects_unknown_keys_and_injection(tmp_path, updates):
    with pytest.raises(ValueError):
        userconfig.write(updates, tmp_path / ".env")

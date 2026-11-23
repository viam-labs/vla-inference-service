from pathlib import Path

import pytest

from vla.policy import installer as ins


def _req_dir(tmp_path: Path) -> Path:
    d = tmp_path / "requirements"
    d.mkdir()
    (d / "base.txt").write_text(
        "torch==2.11.0 ; sys_platform != 'linux'\n"
        "torch==2.11.0+cu128 ; sys_platform == 'linux'\n"
        "torchvision==0.26.0+cu128 ; sys_platform == 'linux'\n"
        "nvidia-cublas-cu12==12.8.4.1 ; sys_platform == 'linux'\n"
        "triton==3.6.0 ; sys_platform == 'linux'\n"
        "lerobot @ git+https://github.com/huggingface/lerobot@abc\n"
        "einops==0.8.1\n"
    )
    for stem in ("smolvla", "evo1", "diffusion", "pi", "cosmos3"):
        (d / f"{stem}.txt").write_text(f"{stem}-dep==1.0\n")
    return d


@pytest.mark.parametrize(
    "policy_type,extra",
    [
        ("smolvla", "smolvla.txt"),
        ("evo1", "evo1.txt"),
        ("diffusion", "diffusion.txt"),
        ("pi0", "pi.txt"),
        ("pi0_fast", "pi.txt"),
        ("pi05", "pi.txt"),
        ("cosmos3_edge", "cosmos3.txt"),
        ("cosmos3_nano", "cosmos3.txt"),
    ],
)
def test_type_maps_to_base_and_its_extra(tmp_path, policy_type, extra):
    d = _req_dir(tmp_path)
    assert ins.requirement_files(d, policy_type) == [d / "base.txt", d / extra]


@pytest.mark.parametrize("policy_type", ["act", "tdmpc", "vqbet"])
def test_base_only_types(tmp_path, policy_type):
    d = _req_dir(tmp_path)
    assert ins.requirement_files(d, policy_type) == [d / "base.txt"]


def test_unlisted_type_is_refused_with_the_supported_list(tmp_path):
    with pytest.raises(ins.InstallError, match="'groot'.*smolvla"):
        ins.requirement_files(_req_dir(tmp_path), "groot")


def test_digest_changes_with_file_contents_and_jetson_settings(tmp_path):
    d = _req_dir(tmp_path)
    files = ins.requirement_files(d, "smolvla")
    a = ins.marker_digest(files, None)
    assert a == ins.marker_digest(files, None)
    assert a != ins.marker_digest(files, {"JETSON_TORCH_VERSION": "2.8.0"})
    (d / "smolvla.txt").write_text("smolvla-dep==2.0\n")
    assert a != ins.marker_digest(files, None)


def test_jetson_filter_drops_torch_and_its_cuda_dependencies():
    text = (
        "torch==2.11.0+cu128 ; sys_platform == 'linux'\n"
        "torchvision==0.26.0+cu128 ; sys_platform == 'linux'\n"
        "nvidia-cublas-cu12==12.8.4.1 ; sys_platform == 'linux'\n"
        "triton==3.6.0 ; sys_platform == 'linux'\n"
        "cuda-toolkit==12.8.1 ; sys_platform == 'linux'\n"
        "cuda-bindings==12.9.4 ; sys_platform == 'linux'\n"
        "lerobot @ git+https://x@abc\n"
        "einops==0.8.1\n"
    )
    assert ins.filter_for_jetson(text) == "lerobot @ git+https://x@abc\neinops==0.8.1\n"


def test_commands_without_jetson_install_all_files_in_one_call(tmp_path):
    d = _req_dir(tmp_path)
    files = ins.requirement_files(d, "smolvla")
    cmds = ins.install_commands("uv", "/py", files, None, tmp_path)
    assert cmds == [["uv", "pip", "install", "--python", "/py", "-r", str(d / "base.txt"), "-r", str(d / "smolvla.txt")]]


def test_commands_on_jetson_install_jetson_torch_first_then_filtered_base(tmp_path):
    d = _req_dir(tmp_path)
    files = ins.requirement_files(d, "smolvla")
    jetson = {"JETSON_TORCH_INDEX": "https://j/idx", "JETSON_TORCH_VERSION": "2.8.0", "JETSON_TORCHVISION_VERSION": "0.23.0"}
    cmds = ins.install_commands("uv", "/py", files, jetson, tmp_path)
    assert cmds[0] == [
        "uv", "pip", "install", "--python", "/py", "--no-deps",
        "--reinstall-package", "torch", "--reinstall-package", "torchvision",
        "--index-url", "https://j/idx", "torch==2.8.0", "torchvision==0.23.0",
    ]
    filtered = tmp_path / "base.jetson.txt"
    assert cmds[1] == ["uv", "pip", "install", "--python", "/py", "-r", str(filtered), "-r", str(d / "smolvla.txt")]
    text = filtered.read_text()
    assert "torch" not in text.replace("lerobot", "") and "nvidia" not in text
    assert "triton" not in text and "cuda" not in text


def test_jetson_settings_none_off_tegra(monkeypatch):
    monkeypatch.setattr(ins, "is_jetson", lambda: False)
    assert ins.jetson_settings({}) is None


def test_jetson_settings_defaults_and_overrides(monkeypatch):
    monkeypatch.setattr(ins, "is_jetson", lambda: True)
    monkeypatch.setattr(ins, "glibc_version", lambda: (2, 39))
    s = ins.jetson_settings({"JETSON_TORCH_VERSION": "2.9.0"})
    assert s == {
        "JETSON_TORCH_INDEX": "https://pypi.jetson-ai-lab.io/jp6/cu129",
        "JETSON_TORCH_VERSION": "2.9.0",
        "JETSON_TORCHVISION_VERSION": "0.23.0",
    }


def test_jetson_with_old_glibc_falls_back_to_generic_with_a_warning(monkeypatch, caplog):
    monkeypatch.setattr(ins, "is_jetson", lambda: True)
    monkeypatch.setattr(ins, "glibc_version", lambda: (2, 35))
    assert ins.jetson_settings({}) is None
    assert "glibc is 2.35" in caplog.text
    assert "JetPack 7" in caplog.text


def test_glibc_version_parses_major_minor_ignoring_suffix(monkeypatch):
    monkeypatch.setattr(ins.platform, "libc_ver", lambda: ("glibc", "2.38a"))
    assert ins.glibc_version() == (2, 38)


@pytest.mark.parametrize("libc", [("glibc", ""), ("", "")])
def test_glibc_version_none_when_unknown(monkeypatch, libc):
    monkeypatch.setattr(ins.platform, "libc_ver", lambda: libc)
    assert ins.glibc_version() is None


def test_jetson_with_unknown_glibc_takes_the_sm87_path(monkeypatch):
    monkeypatch.setattr(ins, "is_jetson", lambda: True)
    monkeypatch.setattr(ins, "glibc_version", lambda: None)
    assert ins.jetson_settings({}) == ins.JETSON_DEFAULTS

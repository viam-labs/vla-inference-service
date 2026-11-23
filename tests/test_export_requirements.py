import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import export_requirements as er  # noqa: E402


def test_package_name_normalizes_specifier_and_url_lines():
    assert er.package_name("torch==2.11.0+cu128 ; sys_platform == 'linux'") == "torch"
    assert er.package_name("lerobot @ git+https://github.com/huggingface/lerobot@abc") == "lerobot"
    assert er.package_name("Typing_Extensions==4.16.0") == "typing-extensions"
    assert er.package_name("nvidia.cublas-cu12==1") == "nvidia-cublas-cu12"


def test_subtract_drops_every_line_of_a_named_package():
    lines = [
        "torch==2.11.0 ; sys_platform != 'linux'",
        "torch==2.11.0+cu128 ; sys_platform == 'linux'",
        "num2words==0.5.14",
    ]
    assert er.subtract(lines, {"torch"}) == ["num2words==0.5.14"]


def test_requirement_lines_skip_comments_blank_and_continuations():
    text = "# header\n\ntorch==2.11.0\n    # via lerobot\nnumpy==2.2.6\n"
    assert er.requirement_lines(text) == ["torch==2.11.0", "numpy==2.2.6"]


def test_checked_subtract_refuses_marker_mismatch():
    with pytest.raises(ValueError, match="smolvla: foo differs"):
        er.checked_subtract(["foo==1.0", "bar==2"], ["foo==1.0 ; sys_platform == 'linux'"], "smolvla")


def test_checked_subtract_passes_when_lines_match():
    kept = ["foo==1.0 ; sys_platform == 'linux'"]
    assert er.checked_subtract([kept[0], "bar==2"], kept, "smolvla") == ["bar==2"]


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_export_against_the_real_lock(tmp_path):
    er.export(tmp_path, project=ROOT)
    files = {p.stem: p for p in tmp_path.glob("*.txt")}
    assert set(files) == {"core", "base", *er.TYPE_EXTRAS}
    names = {stem: {er.package_name(l) for l in er.requirement_lines(p.read_text())} for stem, p in files.items()}
    # An install of core + base + one type never pins a package twice. Type files
    # may share packages with each other (smolvla and pi both carry transformers);
    # only one type is ever installed per load.
    shared = names["core"] | names["base"]
    assert not names["core"] & names["base"]
    for t in er.TYPE_EXTRAS:
        assert not names[t] & shared, (t, names[t] & shared)
    assert {"viam-sdk", "numpy", "huggingface-hub", "uv"} <= names["core"]
    assert "torch" not in names["core"]
    assert {"torch", "torchvision", "lerobot"} <= names["base"]
    assert "num2words" in names["smolvla"]
    assert "diffusers" in names["diffusion"]
    for p in files.values():
        assert "--index-url" not in p.read_text()
        assert "--extra-index-url" not in p.read_text()

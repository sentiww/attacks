import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fake_python(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "arguments.txt"
    python = bin_dir / "python"
    python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE\"\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CAPTURE": str(capture),
    }
    return capture, environment


def test_train_script_forwards_default_configs_and_overrides(tmp_path: Path) -> None:
    capture, environment = _fake_python(tmp_path)

    subprocess.run(
        [PROJECT_ROOT / "scripts" / "train.sh", "configs/extra.yaml"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    assert capture.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "training.train",
        "--config",
        "configs/base.yaml",
        "configs/datasets/cifar10.yaml",
        "configs/train_scratch.yaml",
        "configs/triggers/modes/source_to_target/medium.yaml",
        "configs/triggers/specs/gradient/medium.yaml",
        "configs/extra.yaml",
    ]


def test_preview_script_requires_config_and_forwards_options(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "preview.sh"
    missing = subprocess.run(
        [script],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing.returncode == 2
    assert "Usage:" in missing.stderr

    capture, environment = _fake_python(tmp_path)
    subprocess.run(
        [script, "configs/base.yaml", "--index", "2"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "tools.preview",
        "--config",
        "configs/base.yaml",
        "--index",
        "2",
    ]

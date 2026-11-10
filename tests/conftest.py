import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def demo_split(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo_split")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_data.py"), "--out", str(out)], check=True, cwd=ROOT)
    return out


@pytest.fixture(scope="session")
def sandbox():
    from datamate.sandbox import SandboxClient

    sb = SandboxClient()
    yield sb
    sb.close()

"""pytest 根配置：将仓库根加入 sys.path，保证 core/pipeline/mia/data 可导入。

作者：晨星
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

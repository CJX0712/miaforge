"""MiaForge 错误码体系（E1xx~E5xx），所有异常携带稳定 code。

作者：晨星
"""


class MiaError(Exception):
    """所有 MiaForge 异常的基类。"""

    code = "E000"

    def __init__(self, message: str, code: str | None = None):
        self.code = code or self.code
        super().__init__(f"[{self.code}] {message}")


class DataError(MiaError):
    code = "E100"


class TargetError(MiaError):
    code = "E200"


class AttackError(MiaError):
    code = "E300"


class PipelineError(MiaError):
    code = "E400"


class ConfigError(MiaError):
    code = "E500"

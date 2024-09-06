from typing import Final


class MetaColumns:
    """Metadata columns."""

    OS_M: Final[str] = "OS_M"
    OS_EVENT: Final[str] = "OS_EVENT"
    STAGE_3_PLUS: Final[str] = "STAGE_3+"
    RELAPSE: Final[str] = "RELAPSE"
    SURVIVAL: Final[str] = "SURVIVAL"
    CASE_UUID: Final[str] = "caseUuid"


class ResultsColumns:
    """Results columns."""

    OVERRIDES: Final[str] = "overrides"

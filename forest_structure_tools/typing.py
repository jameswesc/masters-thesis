from typing import Literal, Optional

Op = Literal["at", "above", "below", "above_inc", "below_inc", "inside", "inside_inc"]
PercentageConfig = tuple[Op, int, Optional[int]]
Percentages = list[PercentageConfig]

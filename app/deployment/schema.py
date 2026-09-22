from dataclasses import dataclass, fields
from typing import Any, Literal, overload, Self


DictFields = Literal[
    "l3attack_origin_bitrate",
    "l3attack_origin_duration",
    "l3attack_origin_protocol",
]

ListFields = Literal[
    "httpreq",
    "traffic",
    "aibots_crawlers",
    "bots",
    "l7attack",
    "httpreq_automated",
    "httpreq_human",
    "l3attack_origin",
    "l3attack_target",
    "timestamps",
]

@dataclass(slots=True)
class RegionTimeseriesFetchResult:
    """
    Represents an immutable fetch result object for inference storing 
    timeseries data for a single country. Used for internal data-to-features 
    conversion in memory.
    """

    httpreq: list[str]
    traffic: list[str]
    aibots_crawlers: list[str]
    bots: list[str]
    l7: list[str]
    httpreq_automated: list[str]
    httpreq_human: list[str]
    l3_origin: list[str]
    l3_target: list[str]
    l3_origin_bitrate: dict[str, list[str]]
    l3_origin_duration: dict[str, list[str]]
    l3_origin_protocol: dict[str, list[str]]
    timestamps: list[str]

    @classmethod
    def __getatts__(cls) -> set[str]:
        return {f.name for f in fields(cls)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        required_fields = cls.__getatts__()
        provided_fields = set(data.keys())

        if required_fields != provided_fields:
            raise ValueError(f"[ERROR] Missing required keys: {required_fields ^ provided_fields}")

        kwargs = {k: data[k] for k in required_fields}
        return cls(**kwargs)

    @overload
    def __getitem__(self, key: DictFields) -> dict[str, list[str]]:
        ...

    @overload
    def __getitem__(self, key: ListFields) -> list[str]:
        ...

    def __getitem__(self, key: str) -> dict[str, list[str]] | list[str]:
        if hasattr(self, key):
            val = getattr(self, key)
            if isinstance(val, (dict, list)):
                return val
        raise KeyError(f"[ERROR] {key} not found in fetch results")

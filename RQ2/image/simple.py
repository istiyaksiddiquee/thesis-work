import typing

from flytekit import task, workflow
from typing import Tuple


@task
def t1(a: int) -> typing.NamedTuple("OutputsBC", t1_int_output=int, c=str):
    return a + 2, "world"


@task
def t2(a: str, b: str) -> str:
    return b + a


@workflow
def my_wf(a: int, b: str) -> Tuple[int, str]:
    a = 5
    b = "hello "
    x, y = t1(a=a)
    d = t2(a=y, b=b)
    return x, d
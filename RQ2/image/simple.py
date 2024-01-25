from flytekit import task, workflow
from flytekit.experimental import eager
import asyncio
from flytekit.remote import FlyteRemote
from flytekit.configuration import (
    Config,
    PlatformConfig,
    ImageConfig,
    DataConfig,
    S3Config,
)

@task
def add_one(x: int) -> int:
    return x + 1


@task
def double(x: int) -> int:
    return x * 2


@eager(
    remote = FlyteRemote(
        config=Config(
            platform=PlatformConfig(
                endpoint="127.0.0.1:8080",
                insecure=True,
                insecure_skip_verify=False,
            ),
            data_config=DataConfig(
                s3=S3Config(
                    endpoint="127.0.0.1:9000",
                    access_key_id="minio",
                    secret_access_key="miniostorage",
                )
            ),
        ),
        default_project="flytesnacks",
        default_domain="development",
    )
)
async def simple_eager_workflow() -> int:
    x = 5
    out = await add_one(x=x)
    if out < 0:
        return -1
    return await double(x=out)
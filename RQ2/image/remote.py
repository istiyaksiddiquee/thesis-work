from flytekit.remote import FlyteRemote
from flytekit.configuration import (
    Config,
    PlatformConfig,
    ImageConfig,
    DataConfig,
    S3Config,
)

if __name__ == "__main__":

    remote = FlyteRemote(
        config=Config(
            platform=PlatformConfig(
                endpoint="192.168.0.104:30080",
                insecure=True,
                insecure_skip_verify=False,
            ),
            data_config=DataConfig(
                s3=S3Config(
                    endpoint="http://192.168.0.104:30002",
                    access_key_id="minio",
                    secret_access_key="miniostorage",
                )
            ),
        ),
        default_project="fakeray",
        default_domain="staging",
    )

    from script import work

    wf = remote.register_script(
        entity=work,
        image_config=ImageConfig.from_images("istiyaksiddiquee/flyte-for-kube:1.0.0"),
        source_path="./",
        module_name="script",
        project="fakeray",
        domain="staging",
        version="v1",
    )

    lp = remote.fetch_launch_plan(project="fakeray", domain="staging", name="script.work")
    print(lp.id.version)

    # # Execute
    execution = remote.execute(
        lp,
        inputs={},
        execution_name="workflow-execution-1",
        project="fakeray",
        domain="staging",
        version="v1",
        wait=True,
        image_config="istiyaksiddiquee/flyte-for-kube:1.0.0",
    )

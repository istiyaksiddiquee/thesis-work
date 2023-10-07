from flytekit import task, workflow, Resources


@task(
    container_image="istiyaksiddiquee/flyte-base-image:6.0.0",
    limits=Resources(mem="3000Mi", cpu="1", ephemeral_storage="3000Mi"),
)
def dummy_task():
    sum = 0
    for i in range(1000):
        sum += i

    return sum


@workflow
def work():

    print("here")
    with open("./test.txt", "w") as file:
        file.write("dummy string to test this")
        file.flush()

    summation = dummy_task()
    return summation


if __name__ == "__main__":
    work()
    # FlyteRemote object is the main entrypoint to API
    
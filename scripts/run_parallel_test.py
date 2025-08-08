from time import sleep

from forest_structure_tools import parallel_utils


def test_func(
    a,
    b,
):
    sleep(1)  # Simulate some processing time
    return f"{a} + {b} = {a + b}"


def run_parallel_test():
    nums = range(1, 50)

    ids = [f"ID{i}" for i in nums]
    args_list = [(i, i + 1) for i in nums]

    parallel_utils.run_process_in_parallel(
        test_func,
        ids,
        args_list,
        max_workers=4,
        task_description="Running test function in parallel",
    )


if __name__ == "__main__":
    run_parallel_test()

from forest_structure_tools import parallel_utils
from time import sleep


def test_funct(a, b, ka=1, kb=2):
    sleep(1)  # Simulate some processing time
    return f"{a} + {b} = {a + b}, {ka} * {kb} = {ka * kb}"


def run_parallel_test():
    nums = range(1, 50)

    args_list = [(f"ID{i}", (i, i + 1), {"ka": i * 10, "kb": i * 10 + 5}) for i in nums]

    results = parallel_utils.run_process_in_parallel(
        test_funct,
        args_list,
        max_workers=4,
        task_description="Running test function in parallel",
    )
    # for result in results:
    # print(result)


if __name__ == "__main__":
    run_parallel_test()

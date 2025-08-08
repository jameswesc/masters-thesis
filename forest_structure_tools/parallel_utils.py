from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, List, TypedDict

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)


class ProcessResult(TypedDict):
    id: str
    result: Any
    error: str | None


def run_process_in_parallel(
    func: Callable,
    ids: list[str],
    args_list: list[tuple[Any]],
    max_workers=1,
    task_description: str = "Parallel process",
) -> List[ProcessResult]:
    """
    Run a function in parallel with a list of arguments. Note wont work in Jupyter notebooks.
    This function also uses a progress bar to show the status of the tasks.

    Parameters:
    - func: The function to run in parallel.
    - ids: A list of identifiers for each process.
    - args_list: A list of function arguments
    - max_workers: The maximum number of worker threads to use. If None, defaults to the number of processors on the machine.

    Returns:
    A list of results from each call to `func`.
    """

    if len(ids) != len(args_list):
        raise ValueError("Length of ids and args_list must match.")

    console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[cyan]{task.completed} / {task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task(task_description, total=len(args_list))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(func, *args): id for id, args in zip(ids, args_list)
            }

            results = []
            for future in as_completed(futures):
                id = futures[future]
                try:
                    result = future.result()
                    results.append(ProcessResult(id=id, result=result, error=None))
                except Exception as e:
                    results.append(ProcessResult(id=id, result=None, error=str(e)))

                progress.advance(task)

    return results

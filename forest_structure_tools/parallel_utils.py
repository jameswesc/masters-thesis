from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Tuple, Dict, Any, TypedDict
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
    args_list: List[Tuple[str, Tuple, Dict[str, Any]]],
    max_workers=1,
    task_description: str = "Parallel process",
) -> List[ProcessResult]:
    """
    Run a function in parallel with a list of arguments. Note wont work in Jupyter notebooks.

    Parameters:
    - func: The function to run in parallel.
    - args_list: A list of tuples like (id: str, pargs: tuple, kwargs: dict)
    - max_workers: The maximum number of worker threads to use. If None, defaults to the number of processors on the machine.

    Returns:
    A list of results from each call to `func`.
    """

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
                executor.submit(func, *args, **kwargs): id
                for id, args, kwargs in args_list
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

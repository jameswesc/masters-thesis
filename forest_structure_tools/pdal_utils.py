import pdal
from . import parallel_utils as pu


def process_pdal_pipeline(pipeline: str):
    pl = pdal.Pipeline(pipeline)
    pl.execute()


def process_pdal_pipelines_parallel(
    ids: list[str],
    pipelines: list[str],
    max_workers=1,
):
    if len(ids) != len(pipelines):
        raise ValueError("Length of ids and pipelines must match.")

    args_list = [(pipeline,) for pipeline in pipelines]

    return pu.run_process_in_parallel(
        process_pdal_pipeline, ids, args_list, max_workers=max_workers
    )

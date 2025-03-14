from joblib import Parallel
from tqdm import tqdm


class ProgressParallel(Parallel):
    def __init__(
        self,
        use_tqdm=True,
        total=None,
        leave=False,
        desc=None,
        file_handler=None,
        *args,
        **kwargs
    ):
        self._use_tqdm = use_tqdm
        self._total = total
        self._leave = leave
        self._desc = desc
        self._file_handler = file_handler
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(
            disable=not self._use_tqdm,
            total=self._total,
            leave=self._leave,
            desc=self._desc,
            file=self._file_handler
        ) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

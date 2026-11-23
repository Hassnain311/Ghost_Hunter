"""Hold one OS-owned lock for the lifetime of the interactive game."""

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import tempfile


@contextmanager
def interactive_instance(path=None):
    if path is None:
        # Windows has a per-user temp directory. Unix temp directories are often shared.
        suffix = "" if os.name == "nt" else f"-{os.getuid()}"
        path = Path(tempfile.gettempdir()) / f"GhostHunter{suffix}.lock"
    with open(path, "a+b") as lock:
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            acquired = False
        else:
            acquired = True
        # Closing the handle also releases the lock after an exception or process exit.
        yield acquired

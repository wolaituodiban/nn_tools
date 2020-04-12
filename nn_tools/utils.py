# 尽量使用tensor类型传递数据，不保证numpy类型能正确运行
from functools import wraps

from tqdm import tqdm as _tqdm


@wraps(_tqdm)
def tqdm(*args, **kwargs):
    with _tqdm(*args, **kwargs) as t:
        try:
            for _ in t:
                yield _
        except KeyboardInterrupt:
            t.close()
            raise KeyboardInterrupt

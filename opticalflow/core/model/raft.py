from typing import Any

from .base_model import BaseModel, ModelMode
from .external import raft


class RAFT(BaseModel):

    def __init__(self, args, mode: ModelMode = ModelMode.TEST):
        super().__init__(mode=mode)
        self._model = raft.RAFT(args)

    def _preprocess(self, x: Any):
        if isinstance(x, (tuple, list)):
            x = x[0]
        return x

    def _forward_test(self, x: Any):
        self._model.eval()
        return self._model(x)

    def _forward_train(self, x: Any):
        return self._model(x)

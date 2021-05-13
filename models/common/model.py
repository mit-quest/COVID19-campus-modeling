class Model:
    config = dict()
    model_parameters = dict()

    def __str__(self) -> str:
        return ' | '.join([self.config['model_type'], self.config['model_name'], self.config['model_id']])

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_parameters = self.config['model_parameters']

    def sample(self,  t_0: str, n_samples: int, dates: list, input_samples: dict) -> dict:
        raise NotImplementedError

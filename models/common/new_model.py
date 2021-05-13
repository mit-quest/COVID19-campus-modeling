from models.common.model import Model


class NewModel(Model):
    # This function needs to be specified
    def sample(self,  t_0: str, n_samples: int, dates: list, input_samples: dict) -> dict:
        return dict(dates=dates, samples=[])

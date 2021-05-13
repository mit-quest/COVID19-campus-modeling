import pandas as pd


def add_empty_buildings(missing_list: list, dict_to_add_to: dict):
    '''
    function adds missing buildings (with empty dict as value) to a dict over buildings
    so that jsons being saved always have all buildings
    '''

    for building in missing_list:
        dict_to_add_to[building] = {}
    return dict_to_add_to


def sort_dict(d):
    items = [[k, v] for k, v in sorted(d.items(), key=lambda x: x[0])]
    for item in items:
        if isinstance(item[1], dict):
            item[1] = sort_dict(item[1])
    return dict(items)


class Analysis():
    config = dict()
    analysis_parameters = dict()

    def __str__(self):
        return ' | '.join([self.config['analysis_name'], self.config['analysis_id']])

    def __init__(self, config):
        self.config = config
        self.analysis_parameters = self.config['analysis_parameters']

    def run(self, input_samples: dict, input_analyses: dict) -> dict:
        raise NotImplementedError

    def samples_dict_to_df(self, model_type: str, samples: dict, dates: list = None) -> pd.DataFrame:
        if model_type == 'person':
            df = pd.DataFrame(samples)
        elif model_type == 'immunity':
            df = pd.DataFrame(samples, columns=['immunity'])
        elif model_type == 'prevalence':
            df = pd.concat([pd.DataFrame(samples[i], index=dates)
                            for i in range(len(samples))],
                           keys=range(len(samples)))
            df.columns = ['prevalence-' + c for c in df.columns]
        elif model_type == 'action':
            df = pd.concat([pd.DataFrame(samples[i], index=dates)
                            for i in range(len(samples))],
                           keys=range(len(samples)))
            df.columns = ['action-' + c for c in df.columns]
        elif model_type == 'infection':
            df = pd.concat([pd.DataFrame(samples[i], index=dates, columns=['infection'])
                            for i in range(len(samples))],
                           keys=range(len(samples)))
        elif model_type == 'health-harm':
            df = pd.concat([pd.DataFrame(samples[i], index=dates)
                            for i in range(len(samples))],
                           keys=range(len(samples)))
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
        return df

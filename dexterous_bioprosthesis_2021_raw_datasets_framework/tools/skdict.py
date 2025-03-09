class skdict(dict):
    """
    Dictionary class that supports sklearn methods get_params and set_params
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_params(self, deep=True):
        return dict(self)

    def set_params(self, **params):

        for k, v in params.items():
            self[k] = v

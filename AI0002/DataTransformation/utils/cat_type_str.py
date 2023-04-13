import pandas as pd
class cat_to_str:
    def __init__(self):
        pass

    def convert(self, x):
        return pd.DataFrame(x).astype(str)

        
from monai.data import CacheDataset

class CustomCacheDataSet(CacheDataset):
    '''
        __getitem__() override to ensure the correct return type - dict
    '''

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        if type(data) is list:
            data = data[0]

        return data
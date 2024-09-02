import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_model(train_dataset, config):
    if config.framework == 'vm':
        if config.method == 'c2c_vanilla':
            from models.vm_models.c2c import C2C
            model = C2C(train_dataset, config)
            return model
        elif config.method == 'c2c_enhance':
            from models.vm_models.c2c import C2C
            model = C2C(train_dataset, config)
            return model
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


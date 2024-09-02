import torch

from timesead.models.prediction import Varade, VaradeAnomalyDetector, VaradeLoss
from timesead.optim.trainer import EarlyStoppingHook
from timesead_experiments.utils import data_ingredient, load_dataset, training_ingredient, train_model, make_experiment, \
    make_experiment_tempfile, serialization_guard, get_dataloader
from timesead.utils.utils import Bunch


experiment = make_experiment(ingredients=[data_ingredient, training_ingredient])


def get_training_pipeline():
    return {
        'prediction': {'class': 'PredictionTargetTransform', 'args': {'window_size': 128, 'prediction_horizon': 1,
                                                                      'replace_labels': True}}
    }


def get_test_pipeline():
    return {
        'prediction': {'class': 'PredictionTargetTransform', 'args': {'window_size': 128, 'prediction_horizon': 1,
                                                                      'replace_labels': False}}
    }


def get_batch_dim():
    return 0


@data_ingredient.config
def data_config():
    pipeline = get_training_pipeline()

    ds_args = dict(
        training=True,
    )

    split = (0.75, 0.25)


@training_ingredient.config
def training_config():
    batch_dim = get_batch_dim()
    loss = torch.nn.L1Loss
    trainer_hooks = [
        ('post_validation', EarlyStoppingHook)
    ]


@experiment.config
def config():
    # Model-specific parameters
    model_params = dict()
    train_detector = True
    save_detector = True


@experiment.command(unobserved=True)
@serialization_guard
def get_datasets():
    train_ds, val_ds = load_dataset()

    return get_dataloader(train_ds), get_dataloader(val_ds)


@experiment.command(unobserved=True)
@serialization_guard('model', 'val_loader')
def get_anomaly_detector(model, val_loader, training, _run, save_detector=True):
    training = Bunch(training)
    detector = VaradeAnomalyDetector(model).to(training.device)
    # detector.fit(val_loader)

    if save_detector:
        with make_experiment_tempfile('final_model.pth', _run, mode='wb') as f:
            torch.save(dict(detector=detector), f)

    return detector


@experiment.automain
@serialization_guard
def main(model_params, dataset, training, _run, train_detector=True):
    ds_params = Bunch(dataset)
    train_params = Bunch(training)

    train_ds, val_ds = load_dataset()
    model = Varade(filters=64,channels=train_ds.num_features, inputLength=train_ds.seq_len)

    loss=[VaradeLoss(model)]
    trainer = train_model(_run, model, train_ds, val_ds,loss=loss)
    early_stop = trainer.hooks['post_validation'][-1]
    model = early_stop.load_best_model(trainer, model, train_params.epochs)

    if train_detector:
        detector = get_anomaly_detector(model, trainer.val_iter)
    else:
        detector = None

    return dict(detector=detector, model=model)


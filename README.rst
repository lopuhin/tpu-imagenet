ImageNet on Kaggle TPU
======================

Preparing datasts
-----------------

Note: you can skip this step if you are fine with using datasets already
prepared on Kaggle.

You need imagenet data, with ``train`` and ``val`` folders, each containing
sub-folders corresponding to classes.

Example command to prepare records resized to 320 px on the largest size,
split across 64 files::

    ./prepare_tfrecords.py \
        --max-size 320 \
        --n-shards 64 \
        ./data/imagenet/ \
        ./data/imagenet-tfrec-320

You can use kaggle API to upload the files to datasets, you need to use
several of them, because current dataset size limit is 20 GB.

You also need to use public datasets in order to use
``KaggleDatasets().get_gcs_path()``, which gives you a path to GCS
bucket where your dataset is stored, which TPU can read from directly.

Training
--------

Example training command, assuming Kaggle notebook environment with TPU enabled,
(to be updated to use multiple datasets)::

    from pathlib import Path
    from kaggle_datasets import KaggleDatasets
    gcs_paths = [KaggleDatasets().get_gcs_path(p.name)
                 for p in Path('/kaggle/input/').iterdir()]
    ! train.py {gcs_path}/imagenet-tfrec-320

You can check exact dataset paths with ``! gsutil ls SOME_PATH``,
in case your tfrec files are not on the top level.

Note that if training on your own dataset, you'll need to adjust
``--n-classes`` and ``--n-train-samples``.

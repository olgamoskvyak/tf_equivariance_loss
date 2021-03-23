# Transformation equivariance loss
Implementation of transformation equivariance loss (also can be call transformation
consistency loss).

The loss is used in our paper [Semi-supervised Keypoint Localization](https://openreview.net/pdf?id=yFJ67zTeI2) by Olga Moskvyak, Frederic Maire, Feras Dayoub and Mahsa Baktashmotlagh accepted to ICLR 2021.


## Overview
that ensures that the output of the model stays consisten with the changes of the input.
The loss can be applied

Our method for a semi-supervised method for keypoint localization learns keypoint heatmaps and semantic keypoint representations simultaneously. The model learns from a subset of labeled images and a set of unlabeled images. The method is the most benefitial in low data regimes.

The model is optimized with a supervised loss on the labeled subset and three unsupervised constraints:

![Transformation equivariance idea](images/tf_equivariance.png)


We use TE loss for semi-supervised keypoint localization and observe that TE loss alone gave minor imporvements ... . In our work we complement the loss with learning additional representations.

See the [notebook](https://github.com/olgamoskvyak/tf_equivariance_loss/Examples.ipynb) for the details and examples.



## Usage
TE loss is best used in combination with other supervised and unsupervised losses.
It will not give good results if used alone on unlabeled data.

We use TE loss in our work for semi-supervised keypoint localization to learn
keypoint heatmaps on the unlabeled subset in combination with the supervised loss on
the labeled subset.

Similar losses are used in the following works:
* ELT loss (no implementation)
* [SCOPS: Self-Supervised Co-Part Segmentation](https://varunjampani.github.io/papers/hung19_SCOPS.pdf) implements Equivariance loss with TPS transformations ([code](https://github.com/NVlabs/SCOPS)).


## Examples
```
from tf_equivariance_loss import TfEquivarianceLoss

import torch
import torch.nn as nn

x = torch.rand((4, 3, 64, 64))
model = nn.Sequential(nn.Conv2d(3, 1, 1, bias=False))

tf_equiv_loss = TfEquivarianceLoss(
    transform_type='rotation',
    consistency_type='mse',
    batch_size=4,
    max_angle=90,
    input_hw=(64, 64)
)

# Generate a transformation
tf_equiv_loss.set_tf_matrices()

# Compute model on input image
fx = model(x)

# Transform output
tfx = tf_equiv_loss.transform(fx)

# Transform input image
tx = tf_equiv_loss.transform(x)

# Compute model on the transformed image
ftx = model(tx)

loss = tf_equiv_loss(tfx, ftx)

```

See the [notebook](https://github.com/olgamoskvyak/tf_equivariance_loss/Examples.ipynb) for examples on real data on segmentation model.


## Dependency

* Python >= 3.7
* PyTorch >= 1.5
* Torchvision >= 0.8
* [Kornia](https://github.com/kornia/kornia) = 0.3 (a differentiable computer vision library for PyTorch)







## Citation
Please consider citing our paper if you find this code useful for your research.
```
@inproceedings{ICLR_2021_Moskvyak,
	title = {Semi-supervised Keypoint Localization},
	author = {Olga Moskvyak, Frederic Maire, Feras Dayoub and Mahsa Baktashmotlagh},
	booktitle = {Proc. International Conference on Learning Representations (ICLR)},
	year = {2021}
}
```
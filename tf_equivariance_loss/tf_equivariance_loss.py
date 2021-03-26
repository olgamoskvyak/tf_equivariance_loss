import torch
import torch.nn as nn
import kornia
from kornia.geometry.transform import Resize
import random


class TfEquivarianceLoss(nn.Module):
    """ Transformation Equivariance (TE) Loss.

    TE loss penalizes the difference between output of the model on the
    transformed image and a transformed model's output of the original image.
    TE loss utilizes differentiable transformations from Kornia library.

    Input:
        transform_type (str): type of transformation.
            Implemented types: ['rotation'].
            Default: 'rotation'
        consistency_type (str): difference function in the image space.
            Implemented types : ['mse']'
            Default: 'mse'
        batch_size (int): expected size of the batch. Default: 32.
        max_angle (int): maximum angle of rotation for rotation transformation.
            Default: 90.
        input_hw (tuple of int): image size (height, width).
            Default: (256, 128).

    Example:
        >>> # ENABLE_DOCTEST
        >>> from tf_equivariance_loss import TfEquivarianceLoss
        >>> import torch
        >>> import torch.nn as nn
        >>> x = torch.rand((4, 3, 64, 64))
        >>> model = nn.Sequential(nn.Conv2d(3, 1, 1, bias=False))
        >>> tf_equiv_loss = TfEquivarianceLoss(
        >>>     transform_type='rotation',
        >>>     consistency_type='mse',
        >>>     batch_size=4,
        >>>     max_angle=90,
        >>>     input_hw=(64, 64)
        >>>     )
        >>> # Generate a transformation and compute loss
        >>> tf_equiv_loss.set_tf_matrices()
        >>> fx = model(x)
        >>> tfx = tf_equiv_loss.transform(fx)
        >>> tx = tf_equiv_loss.transform(x)
        >>> ftx = model(tx)
        >>> loss = tf_equiv_loss(tfx, ftx)
        >>> assert loss < 1e-5

    """
    def __init__(self,
                 transform_type='rotation',
                 consistency_type='mse',
                 batch_size=32,
                 max_angle=90,
                 input_hw=(256, 128),
                 ):
        super(TfEquivarianceLoss, self).__init__()
        self.transform_type = transform_type
        self.batch_size = batch_size
        self.max_angle = max_angle
        self.input_hw = input_hw

        if consistency_type == 'mse':
            self.consistency = nn.MSELoss()
        elif consistency_type == 'bse':
            self.consistency = nn.BSELoss()
        else:
            raise ValueError('Incorrect consistency_type {}'.
                             format(consistency_type))

        self.tf_matrices = None

    def set_tf_matrices(self):
        """ Set transformation matrices
        """
        if self.transform_type == 'rotation':
            self.tf_matrices = self._get_rotation()

    def _get_rotation(self):
        """ Get transformation matrices

        Output:
            rot_mat (float torch.Tensor): tensor of shape (batch_size, 2, 3)
        """
        # define the rotation center
        center = torch.ones(self.batch_size, 2)
        center[..., 0] = self.input_hw[1] / 2  # x
        center[..., 1] = self.input_hw[0] / 2  # y

        # create transformation (rotation)
        angle = torch.tensor(
            [random.randint(-self.max_angle, self.max_angle)
             for _ in range(self.batch_size)]
        )
        # define the scale factor
        scale = torch.ones(self.batch_size)

        # compute the transformation matrix
        tf_matrices = kornia.get_rotation_matrix2d(center, angle, scale)
        return tf_matrices

    def transform(self, x):
        """ Transform input with transformation matrices
        Input:
            x (float torch.Tensor): input data of shape (batch_size, ch, h, w)

        Output:
            xf_x (float torch.Tensor): transformed data
                of shape (batch_size, ch, h, w)

        If transformation input is different from the input size defined
        in the construction then resize the input before transformation.
        Input size is important because transformation matrices depend on
        the image size, e.g the center of rotation.
        """
        resize_input = False

        # Check input size
        if x.shape[2:] != self.input_hw:
            curr_hw = x.shape[2:]
            x = Resize(self.input_hw)(x)
            resize_input = True

        # Transform image
        tf_x = kornia.warp_affine(
            x.float(),
            self.tf_matrices,
            dsize=self.input_hw)

        # Transform back if image has been resized
        if resize_input:
            tf_x = Resize(curr_hw)(tf_x)
        return tf_x

    def forward(self, tfx, ftx):
        loss = self.consistency(tfx, ftx)
        return loss

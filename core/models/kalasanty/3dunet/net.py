import numpy as np
import h5py

import pybel

import tfbio.net
import tfbio.data

from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing

from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D, concatenate
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2

from .data import DataWrapper, get_box_size

__all__ = [
    'dice',
    'dice_np',
    'dice_loss',
    'ovl',
    'ovl_np',
    'ovl_loss',
    'UNet',
]


def dice(y_true, y_pred, smoothing_factor=0.01):
    """Dice coefficient adapted for continuous data (predictions) computed with
    keras layers.
    """

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ((2. * intersection + smoothing_factor)
            / (K.sum(y_true_f) + K.sum(y_pred_f) + smoothing_factor))


def dice_np(y_true, y_pred, smoothing_factor=0.01):
    """Dice coefficient adapted for continuous data (predictions) computed with
    numpy arrays.
    """

    intersection = (y_true * y_pred).sum()
    total = y_true.sum() + y_pred.sum()

    return ((2. * intersection + smoothing_factor)
            / (total + smoothing_factor))


def dice_loss(y_true, y_pred):
    """Keras loss function for Dice coefficient (loss(t, y) = -dice(t, y))"""
    return -dice(y_true, y_pred)


def ovl(y_true, y_pred, smoothing_factor=0.01):
    """Overlap coefficient computed with keras layers"""
    concat = K.concatenate((y_true, y_pred))
    return ((K.sum(K.min(concat, axis=-1)) + smoothing_factor)
            / (K.sum(K.max(concat, axis=-1)) + smoothing_factor))


def ovl_np(y_true, y_pred, smoothing_factor=0.01):
    """Overlap coefficient computed with numpy arrays"""
    concat = np.concatenate((y_true, y_pred), axis=-1)
    return ((concat.min(axis=-1).sum() + smoothing_factor)
            / (concat.max(axis=-1).sum() + smoothing_factor))


def ovl_loss(y_true, y_pred):
    """Keras loss function for overlap coefficient (loss(t, y) = -ovl(t, y))"""
    return -ovl(y_true, y_pred)


class UNet(Model):
    """3D Convolutional neural network based on U-Net
    see: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
         https://arxiv.org/abs/1505.04597
    """
    DEFAULT_SIZE = 36

    def __init__(self, inputs=None, outputs=None, data_handle=None,
                 featurizer=None, scale=None, box_size=None, input_channels=None,
                 output_channels=None, l2_lambda=1e-3, **kwargs):
        """Creates a new network. The model can be either initialized from
        `inputs` and `outputs` (keras layers), `data_handle` (DataWrapper
        object, from which all the shapes are inferred) or manually using
        `box_size`, `input_channels` and `output_channels` arguments. L2
        regularization is used (controlled by `l2_lambda` parameter) and all
        other arguments are passed to keras Model constructor.
        """

        if data_handle is not None:
            if not isinstance(data_handle, DataWrapper):
                raise TypeError('data_handle should be a DataWrapper object,'
                                ' got %s instead' % type(data_handle))

            if box_size is None:
                box_size = data_handle.box_size
            elif box_size != data_handle.box_size:
                raise ValueError('specified box_size does not match '
                                 'data_handle.box_size (%s != %s)'
                                 % (box_size, data_handle.box_size))

            if input_channels is None:
                input_channels = data_handle.x_channels
            elif input_channels != data_handle.x_channels:
                raise ValueError('specified input_channels does not match '
                                 'data_handle.x_channels (%s != %s)'
                                 % (input_channels, data_handle.x_channels))

            if output_channels is None:
                output_channels = data_handle.y_channels
            elif output_channels != data_handle.y_channels:
                raise ValueError('specified output_channels does not match '
                                 'data_handle.y_channels (%s != %s)'
                                 % (output_channels, data_handle.y_channels))
            if scale is None:
                self.scale = data_handle.scale
            elif scale != data_handle.scale:
                raise ValueError('specified scale does not match '
                                 'data_handle.scale (%s != %s)'
                                 % (scale, data_handle.scale))
            self.max_dist = data_handle.max_dist
        else:
            self.scale = scale
            self.max_dist = None    # we'll calculate it later from box size

        if featurizer is not None:
            if not isinstance(featurizer, tfbio.data.Featurizer):
                raise TypeError('featurizer should be a tfbio.data.Featurizer '
                                'object, got %s instead' % type(featurizer))
            if input_channels is None:
                input_channels = len(featurizer.FEATURE_NAMES)
            elif input_channels != len(featurizer.FEATURE_NAMES):
                raise ValueError(
                    'specified input_channels or data_handle.x_channels does '
                    'not match number of features produce by featurizer '
                    '(%s != %s)' % (input_channels, len(featurizer.FEATURE_NAMES)))

        if inputs is not None:
            if outputs is None:
                raise ValueError('you must provide both inputs and outputs')
            if isinstance(inputs, list):
                i_shape = UNet.__total_shape(inputs)
            else:
                i_shape = inputs.shape

            if isinstance(outputs, list):
                o_shape = UNet.__total_shape(outputs)
            else:
                o_shape = outputs.shape

            if len(i_shape) != 5:
                raise ValueError('input should be 5D, got %sD instead'
                                 % len(i_shape))
            elif len(o_shape) != 5:
                raise ValueError('output should be 5D, got %sD instead'
                                 % len(o_shape))
            elif i_shape[1:4] != o_shape[1:4]:
                raise ValueError('input and output shapes do not match '
                                 '(%s != %s)' % (i_shape[1:4], o_shape[1:4]))
            if box_size is None:
                box_size = i_shape[1]
            elif i_shape[1:4] != (box_size,) * 3:
                raise ValueError('input shape does not match box_size '
                                 '(%s != %s)' % (i_shape[1:4], (box_size,) * 3))

            if input_channels is not None and i_shape[4] != input_channels:
                raise ValueError('number of channels (specified via featurizer'
                                 ', input_channels or data_handle) does not '
                                 'match input shape (%s != %s)'
                                 % (i_shape[4], input_channels))
            if output_channels is not None and o_shape[4] != output_channels:
                raise ValueError('specified output_channels or '
                                 'data_handle.y_channels does not match '
                                 'output shape (%s != %s)'
                                 % (o_shape[4], output_channels))
        else:
            if outputs is not None:
                raise ValueError('you must provide both inputs and outputs')
            elif (box_size is None or input_channels is None
                  or output_channels is None):
                raise ValueError('you must either provide: 1) inputs and '
                                 'outputs (keras layers); 2) data_handle '
                                 '(DataWrapper object); 3) box_size, '
                                 'input_channels and output_channels')
            elif (box_size < self.DEFAULT_SIZE
                  or box_size % self.DEFAULT_SIZE != 0):
                raise ValueError('box_size does not match the default '
                                 'architecture. Pleas scecify inputs and outputs')

            params = {'kernel_size': 3, 'activation': 'relu',
                      'padding': 'same', 'kernel_regularizer': l2(l2_lambda)}
            inputs = Input((box_size, box_size, box_size, input_channels), name='input')
            conv1 = Convolution3D(filters=32, **params)(inputs)
            conv1 = Convolution3D(filters=32, **params)(conv1)
            pool1 = MaxPooling3D(pool_size=2)(conv1)

            conv2 = Convolution3D(filters=64, **params)(pool1)
            conv2 = Convolution3D(filters=64, **params)(conv2)
            pool2 = MaxPooling3D(pool_size=2)(conv2)

            conv3 = Convolution3D(filters=128, **params)(pool2)
            conv3 = Convolution3D(filters=128, **params)(conv3)
            pool3 = MaxPooling3D(pool_size=3)(conv3)

            conv4 = Convolution3D(filters=256, **params)(pool3)
            conv4 = Convolution3D(filters=256, **params)(conv4)
            pool4 = MaxPooling3D(pool_size=3)(conv4)

            conv5 = Convolution3D(filters=512, **params)(pool4)
            conv5 = Convolution3D(filters=512, **params)(conv5)

            up6 = concatenate([UpSampling3D(size=3)(conv5), conv4], axis=4)
            conv6 = Convolution3D(filters=256, **params)(up6)
            conv6 = Convolution3D(filters=256, **params)(conv6)

            up7 = concatenate([UpSampling3D(size=3)(conv6), conv3], axis=4)
            conv7 = Convolution3D(filters=128, **params)(up7)
            conv7 = Convolution3D(filters=128, **params)(conv7)

            up8 = concatenate([UpSampling3D(size=2)(conv7), conv2], axis=4)
            conv8 = Convolution3D(filters=64, **params)(up8)
            conv8 = Convolution3D(filters=64, **params)(conv8)

            up9 = concatenate([UpSampling3D(size=2)(conv8), conv1], axis=4)
            conv9 = Convolution3D(filters=32, **params)(up9)
            conv9 = Convolution3D(filters=32, **params)(conv9)

            outputs = Convolution3D(
                filters=output_channels,
                kernel_size=1,
                activation='sigmoid',
                kernel_regularizer=l2(l2_lambda),
                name='pocket'
            )(conv9)

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.data_handle = data_handle
        self.featurizer = featurizer
        if self.max_dist is None and self.scale is not None:
            self.max_dist = (box_size - 1) / (2 * self.scale)

    @staticmethod
    def __total_shape(tensor_list):
        if len(tensor_list) == 1:
            total_shape = tuple(tensor_list[0].shape.as_list())
        else:
            total_shape = (*tensor_list[0].shape.as_list()[:-1],
                           sum(t.shape.as_list()[-1] for t in tensor_list))
        return total_shape

    def save_keras(self, path):
        class_name = self.__class__.__name__
        self.__class__.__name__ = 'Model'
        self.save(path, include_optimizer=False)
        self.__class__.__name__ = class_name

    @staticmethod
    def load_model(path, **attrs):
        """Load model saved in HDF format"""
        from keras.models import load_model as keras_load
        custom_objects = {name: val for name, val in globals().items()
                          if name in __all__}
        model = keras_load(path, custom_objects=custom_objects)

        if 'data_handle' in attrs:
            if not isinstance(attrs['data_handle'], DataWrapper):
                raise TypeError('data_handle should be a DataWrapper object, '
                                'got %s instead' % type(attrs['data_handle']))
            elif 'scale' not in attrs:
                attrs['scale'] = attrs['data_handle'].scale
            elif attrs['scale'] != attrs['data_handle'].scale:
                raise ValueError('specified scale does not match '
                                 'data_handle.scale (%s != %s)'
                                 % (attrs['scale'], attrs['data_handle'].scale))

            if 'featurizer' in attrs:
                if not (isinstance(attrs['featurizer'], tfbio.data.Featurizer)):
                    raise TypeError(
                        'featurizer should be a tfbio.data.Featurizer object, '
                        'got %s instead' % type(attrs['featurizer']))
                elif (len(attrs['featurizer'].FEATURE_NAMES)
                      != attrs['data_handle'].x_channels):
                    raise ValueError(
                        'number of features produced be the featurizer does '
                        'not match data_handle.x_channels (%s != %s)'
                        % (len(attrs['featurizer'].FEATURE_NAMES),
                           attrs['data_handle'].x_channels))

            if 'max_dist' not in attrs:
                attrs['max_dist'] = attrs['data_handle'].max_dist
            elif attrs['max_dist'] != attrs['data_handle'].max_dist:
                raise ValueError('specified max_dist does not match '
                                 'data_handle.max_dist (%s != %s)'
                                 % (attrs['max_dist'],
                                    attrs['data_handle'].max_dist))

            if 'box_size' not in attrs:
                attrs['box_size'] = attrs['data_handle'].box_size
            elif attrs['box_size'] != attrs['data_handle'].box_size:
                raise ValueError('specified box_size does not match '
                                 'data_handle.box_size (%s != %s)'
                                 % (attrs['box_size'],
                                    attrs['data_handle'].box_size))

        elif 'featurizer' in attrs and not (isinstance(attrs['featurizer'],
                                            tfbio.data.Featurizer)):
            raise TypeError(
                'featurizer should be a tfbio.data.Featurizer object, '
                'got %s instead' % type(attrs['featurizer']))

        if 'scale' in attrs and 'max_dist' in attrs:
            box_size = get_box_size(attrs['scale'], attrs['max_dist'])
            if 'box_size' in attrs:
                if not attrs['box_size'] == box_size:
                    raise ValueError('specified box_size does not match '
                                     'size defined by scale and max_dist (%s != %s)'
                                     % (attrs['box_size'], box_size))
            else:
                attrs['box_size'] = box_size

        # TODO: add some attrs validation if handle is not specified

        for attr, value in attrs.items():
            setattr(model, attr, value)
        return model

    def pocket_density_from_mol(self, mol):
        """Predict porobability density of pockets using pybel.Molecule object
        as input"""

        if not isinstance(mol, pybel.Molecule):
            raise TypeError('mol should be a pybel.Molecule object, got %s '
                            'instead' % type(mol))
        if self.featurizer is None:
            raise ValueError('featurizer must be set to make predistions for '
                             'molecules')
        if self.scale is None:
            raise ValueError('scale must be set to make predistions')
        prot_coords, prot_features = self.featurizer.get_features(mol)
        centroid = prot_coords.mean(axis=0)
        prot_coords -= centroid
        resolution = 1. / self.scale
        x = tfbio.data.make_grid(prot_coords, prot_features,
                                 max_dist=self.max_dist,
                                 grid_resolution=resolution)
        density = self.predict(x)
        origin = (centroid - self.max_dist)
        step = np.array([1.0 / self.scale] * 3)
        return density, origin, step

    def pocket_density_from_grid(self, pdbid):
        """Predict porobability density of pockets using 3D grid (np.ndarray)
        as input"""

        if self.data_handle is None:
            raise ValueError('data_handle must be set to make predictions '
                             'using PDBIDs')
        if self.scale is None:
            raise ValueError('scale must be set to make predistions')
        x, _ = self.data_handle.prepare_complex(pdbid)
        origin = (self.data_handle[pdbid]['centroid'][:] - self.max_dist)
        step = np.array([1.0 / self.scale] * 3)
        density = self.predict(x)
        return density, origin, step

    def save_density_as_cmap(self, density, origin, step, fname='pockets.cmap',
                             mode='w', name='protein'):
        """Save predcited pocket density as .cmap file (which can be opened in
        UCSF Chimera or ChimeraX)
        """
        if len(density) != 1:
            raise ValueError('saving more than one prediction at a time is not'
                             ' supported')
        density = density[0].transpose([3, 2, 1, 0])

        with h5py.File(fname, mode) as cmap:
            g1 = cmap.create_group('Chimera')
            for i, channel_dens in enumerate(density):
                    g2 = g1.create_group('image%s' % (i + 1))
                    g2.attrs['chimera_map_version'] = 1
                    g2.attrs['name'] = name.encode() + b' binding sites'
                    g2.attrs['origin'] = origin
                    g2.attrs['step'] = step
                    g2.create_dataset('data_zyx', data=channel_dens,
                                      shape=channel_dens.shape,
                                      dtype='float32')

    def save_density_as_cube(self, density, origin, step, fname='pockets.cube',
                             mode='w', name='protein'):
        """Save predcited pocket density as .cube file (format originating from
        Gaussian package).
        """
        angstrom2bohr = 1.889725989

        if len(density) != 1:
            raise ValueError('saving more than one prediction at a time is not'
                             ' supported')
        if density.shape[-1] != 1:
            raise NotImplementedError('saving multichannel density is not'
                                      ' supported yet, please save each'
                                      ' channel in a separate file.')

        with open(fname, 'w') as f:
            f.write('%s CUBE FILE.\n' % name)
            f.write('OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n')
            f.write('    1 %12.6f %12.6f %12.6f\n' % tuple(angstrom2bohr * origin))
            f.write(
                '%5i %12.6f     0.000000      0.000000\n'
                '%5i     0.000000 %12.6f      0.000000\n'
                '%5i     0.000000      0.000000 %12.6f\n'
                % tuple(i for pair in zip(density.shape[1:4],
                                          angstrom2bohr * step) for i in pair)
            )
            f.write('    1     0.000000 %12.6f %12.6f %12.6f\n'
                    % tuple(angstrom2bohr * origin))
            f.write('\n'.join([' '.join('%12.6f' % i for i in row)
                    for row in density.reshape((-1, 6))]))

    def get_pockets_segmentation(self, density, threshold=0.5, min_size=50):
        """Predict pockets using specified threshold on the probability density.
        Filter out pockets smaller than min_size A^3
        """

        if len(density) != 1:
            raise ValueError('segmentation of more than one pocket is not'
                             ' supported')

        voxel_size = (1 / self.scale) ** 3
        # get a general shape, without distinguishing output channels
        bw = closing((density[0] > threshold).any(axis=-1))

        # remove artifacts connected to border
        cleared = clear_border(bw)

        # label regions
        label_image, num_labels = label(cleared, return_num=True)
        for i in range(1, num_labels + 1):
            pocket_idx = (label_image == i)
            pocket_size = pocket_idx.sum() * voxel_size
            if pocket_size < min_size:
                label_image[np.where(pocket_idx)] = 0
        return label_image

    def predict_pocket_atoms(self, mol, dist_cutoff=4.5, expand_residue=True,
                             **pocket_kwargs):
        """Predict pockets for a given molecule and get AAs forming them
        (list pybel.Molecule objects).

        Parameters
        ----------
        mol: pybel.Molecule object
            Protein structure
        dist_cutoff: float, optional (default=2.0)
            Maximal distance between protein atom and predicted pocket
        expand_residue: bool, optional (default=True)
            Inlude whole residue if at least one atom is included in the pocket
        pocket_kwargs:
            Keyword argument passed to `get_pockets_segmentation` method

        Returns
        -------
        pocket_mols: list of pybel.Molecule objects
            Fragments of molecule corresponding to detected pockets.
        """

        from scipy.spatial.distance import cdist

        coords = np.array([a.coords for a in mol.atoms])
        atom2residue = np.array([a.residue.idx for a in mol.atoms])
        residue2atom = np.array([[a.idx - 1 for a in r.atoms]
                                 for r in mol.residues])

        # predcit pockets
        density, origin, step = self.pocket_density_from_mol(mol)
        pockets = self.get_pockets_segmentation(density, **pocket_kwargs)

        # find atoms close to pockets
        pocket_atoms = []
        for pocket_label in range(1, pockets.max() + 1):
            indices = np.argwhere(pockets == pocket_label).astype('float32')
            indices *= step
            indices += origin
            distance = cdist(coords, indices)
            close_atoms = np.where((distance < dist_cutoff).any(axis=1))[0]
            if len(close_atoms) == 0:
                continue
            if expand_residue:
                residue_ids = np.unique(atom2residue[close_atoms])
                close_atoms = np.concatenate(residue2atom[residue_ids])
            pocket_atoms.append([int(idx) for idx in close_atoms])

        # create molecules correcponding to atom indices
        pocket_mols = []
        # TODO optimize (copy atoms to new molecule instead of deleting?)
        for pocket in pocket_atoms:
            # copy molecule
            pocket_mol = mol.clone
            atoms_to_del = (set(range(len(pocket_mol.atoms)))
                            - set(pocket))
            pocket_mol.OBMol.BeginModify()
            for aidx in sorted(atoms_to_del, reverse=True):
                atom = pocket_mol.OBMol.GetAtom(aidx + 1)
                pocket_mol.OBMol.DeleteAtom(atom)
            pocket_mol.OBMol.EndModify()
            pocket_mols.append(pocket_mol)

        return pocket_mols

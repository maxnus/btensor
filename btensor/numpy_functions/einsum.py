import copy
import string

import numpy as np


def einsum(subscripts, *operands, einsumfunc=np.einsum, **kwargs):
    """Allows contraction of Array objects using Einstein notation.

    The overlap matrices between non-matching dimensions are automatically added.
    """
    if '...' in subscripts:
        raise NotImplementedError

    # Remove spaces
    subscripts = subscripts.replace(' ', '')
    if '->' in subscripts:
        labels, result = subscripts.split('->')
    else:
        labels = subscripts
        # Generate result subscripts automatically: all non-repeated subcripts in alphabetical order
        result = ''.join([s for s in sorted(set(subscripts.replace(',', ''))) if labels.count(s) == 1])
    labels = labels.split(',')
    if len(labels) != len(operands):
        raise ValueError("invalid number of operands")

    labels = [list(label) for label in labels]
    labels_out = copy.deepcopy(labels)

    # Sorted set of all indices
    indices = [x for idx in labels for x in idx]
    # Remove duplicates while keeping order (sets do not keep order):
    indices = list(dict.fromkeys(indices).keys())
    # Unused indices
    free_indices = sorted(set(string.ascii_letters).difference(set(indices)))

    # Loop over all indices
    overlaps = []
    basis_dict = {}
    for index in indices:

        # Find smallest basis for given idx:
        basis_target = None
        for i, label in enumerate(labels):
            # The index might appear multiple times per label -> loop over positions
            positions = np.asarray(np.asarray(label) == index).nonzero()[0]
            for pos in positions:
                basis_current = operands[i].basis[pos]
                if basis_target is None or (basis_current.size < basis_target.size):
                    basis_target = basis_current

        assert (basis_target is not None)
        basis_dict[index] = basis_target

        # Replace all other bases corresponding to the same index:
        for i, label in enumerate(labels):
            positions = np.asarray(np.asarray(label) == index).nonzero()[0]
            for pos in positions:
                basis_current = operands[i].basis[pos]
                # If the bases are the same, continue, to avoid inserting an unnecessary identity matrix:
                if basis_current == basis_target:
                    continue
                # Add transformation from basis_current to basis_target:
                index_new = free_indices.pop(0)
                labels_out[i][pos] = index_new
                labels_out.append([index_new, index])
                overlaps.append(basis_current.get_overlap(basis_target).to_numpy(copy=False))

    # Return
    subscripts_out = ','.join([''.join(label) for label in labels_out])
    subscripts_out = '->'.join((subscripts_out, result))
    operands_out = [op.to_numpy(copy=False) for op in operands]
    operands_out.extend(overlaps)
    values = einsumfunc(subscripts_out, *operands_out, **kwargs)
    basis_out = tuple([basis_dict[idx] for idx in result])

    cls = type(operands[0])
    return cls(values, basis_out)

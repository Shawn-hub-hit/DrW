from dataset import *

class Dataloaders(object):
    def __init__(self, dataset, stage='train', batch_size=None,
                pin_memory: bool = False,
                timeout: int = 0,
                num_workers: int = 0,
                worker_init_fn=None,
                fixed_length_left=10,
                fixed_length_right=40):

        self._dataset = dataset
        self._timeout = timeout
        self._num_workers = num_workers
        self._worker_init_fn = worker_init_fn
        self._device = device
        self._stage = stage
        self._pin_momory = pin_memory
        self._fixed_length_left = fixed_length_left
        self._fixed_length_right = fixed_length_right
        self._pad_value = 0
        self._pad_mode = 'pre'
        self.with_histogram = False

        self._dataloader = data.DataLoader(
            self._dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
            batch_sampler=None,
            num_workers=self._num_workers,
            pin_memory=self._pin_momory,
            timeout=self._timeout,
            worker_init_fn=self._worker_init_fn,
        )

    def __len__(self) -> int:
        """Get the total number of batches."""
        return len(self._dataset)

    @property
    def id_left(self) -> np.ndarray:
        """`id_left` getter."""
        x, _ = self._dataset[:]
        return x['id_left']

    @property
    def label(self) -> np.ndarray:
        """`label` getter."""
        _, y = self._dataset[:]
        return y.squeeze() if y is not None else None

    def __iter__(self):
        """Iteration."""
        for batch_data in self._dataloader:
            x, y = batch_data
            self.process_on_batch_data(x, y)

            batch_x = {}
            for key, value in x.items():
                # if key == 'id_right':
                #     continue
                batch_x[key] = torch.tensor(
                    value, device=self._device)

            if self._stage == 'test':
                yield batch_x, None
            else:
                if y.dtype == 'int':  # task='classification'
                    batch_y = torch.tensor(
                        y.squeeze(axis=-1), dtype=torch.long, device=self._device)
                else:  # task='ranking'
                    batch_y = torch.tensor(
                        y, dtype=torch.float, device=self._device)
                yield batch_x, batch_y

    def process_on_batch_data(self, x, y):
        batch_size = len(x['id_left'])
        pad_length_left = max(x['length_left'])
        pad_length_right = max(x['length_right'])
        if self.with_histogram == True:
            bin_size = len(x['match_histogram'][0][0])

        if self._fixed_length_left is not None:
            pad_length_left = self._fixed_length_left
        if self._fixed_length_right is not None:
            pad_length_right = self._fixed_length_right

        for key, value in x.items():
            if key != 'text_left' and key != 'text_right' and \
                    key != 'match_histogram':
                continue

            dtype = self._infer_dtype(value)

            if key == 'text_left':
                padded_value = np.full([batch_size, pad_length_left],
                                       self._pad_value, dtype=dtype)
                self._padding_2D(value, padded_value, self._pad_mode)
            elif key == 'text_right':
                padded_value = np.full([batch_size, pad_length_right],
                                       self._pad_value, dtype=dtype)
                self._padding_2D(value, padded_value, self._pad_mode)
            else:  # key == 'match_histogram'
                padded_value = np.full(
                    [batch_size, pad_length_left, bin_size],
                    self._pad_value, dtype=dtype)
                self._padding_3D(value, padded_value, self._pad_mode)
            x[key] = padded_value

    def _infer_dtype(self, value):
        """Infer the dtype for the features.

        It is required as the input is usually array of objects before padding.
        """
        while isinstance(value, (list, tuple)) and len(value) > 0:
            value = value[0]

        if not isinstance(value, Iterable):
            return np.array(value).dtype

        if value is not None and len(value) > 0 and np.issubdtype(
                np.array(value).dtype, np.generic):
            dtype = np.array(value[0]).dtype
        else:
            dtype = value.dtype

        # Single Precision
        if dtype == np.double:
            dtype = np.float32

        return dtype

    def _padding_2D(self, input, output, mode: str = 'pre'):
        """
        Pad the input 2D-tensor to the output 2D-tensor.

        :param input: The input 2D-tensor contains the origin values.
        :param output: The output is a shapped 2D-tensor which have filled with pad
         value.
        :param mode: The padding model, which can be 'pre' or 'post'.
        """
        batch_size = min(output.shape[0], len(input))
        pad_length = output.shape[1]
        if mode == 'post':
            for i in range(batch_size):
                end_pos = min(len(input[i]), pad_length)
                if end_pos > 0:
                    output[i][:end_pos] = input[i][:end_pos]
        elif mode == 'pre':
            for i in range(batch_size):
                start_pos = min(len(input[i]), pad_length)
                if start_pos > 0:
                    output[i][-start_pos:] = input[i][-start_pos:]
        else:
            raise ValueError('{} is not a vaild pad mode.'.format(mode))

    def _padding_3D(self, input, output, mode: str = 'pre'):
        """
        Pad the input 3D-tensor to the output 3D-tensor.

        :param input: The input 3D-tensor contains the origin values.
        :param output: The output is a shapped 3D-tensor which have filled with pad
         value.
        :param mode: The padding model, which can be 'pre' or 'post'.
        """
        batch_size = min(output.shape[0], len(input))
        pad_1d_length = output.shape[1]
        pad_2d_length = output.shape[2]
        if mode == 'post':
            for i in range(batch_size):
                len_d1 = min(len(input[i]), pad_1d_length)
                for j in range(len_d1):
                    end_pos = min(len(input[i][j]), pad_2d_length)
                    if end_pos > 0:
                        output[i][j][:end_pos] = input[i][j][:end_pos]
        elif mode == 'pre':
            for i in range(batch_size):
                len_d1 = min(len(input[i]), pad_1d_length)
                for j in range(len_d1):
                    start_pos = min(len(input[i][j]), pad_2d_length)
                    if start_pos > 0:
                        output[i][j][-start_pos:] = input[i][j][-start_pos:]
        else:
            raise ValueError('{} is not a vaild pad mode.'.format(mode))
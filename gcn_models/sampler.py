import multiprocessing as mp
from multiprocessing import Queue

import numpy as np
from torch_geometric.data import NeighborSampler


# NOTE: Not filtering positive edges no

class MultiProcessSampler:
    """
    A sampler that uses multiple processes to produce training batches in parallel.
    
    This class leverages Python's multiprocessing to generate batches of training data
    with negative sampling, and can optionally use a masked neighbor sampler.

    Attributes:
        dataset: The dataset containing graph triples and data.
        neg_sample_size (int): The number of negative samples to generate (must be even).
        batch_size (int): The size of each training batch.
        num_procs (int): The number of processes to use for parallel sampling.
        layer_sizes: The sizes of each layer for the neighbor sampling.
        num_epochs (int): The number of epochs to sample (-1 for indefinite).
        masked_sampler (bool): Whether to use a masked neighbor sampler.
        mask: A boolean mask used for the masked neighbor sampler.
        mask_ratio (int): Ratio at which to switch to the masked sampler.
        enforce_type (bool): Whether to enforce type constraints during sampling.
        filter_positive (bool): Whether to filter out positive edges.
    """
    def __init__(self, *, dataset, neg_sample_size, batch_size, num_procs, layer_sizes, num_epochs=-1,
                 masked_sampler: bool = False, mask=None, mask_ratio=0,
                 enforce_type=False,
                 filter_positive=False):
        self.dataset = dataset
        self.neg_sample_size = neg_sample_size
        # Ensure negative sample size is even.
        assert neg_sample_size % 2 == 0
        self.batch_size = batch_size
        # Compute the number of batches per epoch.
        self.len = len(dataset.triples[0]) // batch_size
        self.filter_positive = filter_positive
        self.enforce_type = enforce_type
        self.masked_sampler = masked_sampler
        self.mask_ratio = mask_ratio

        # Initialize the standard neighbor sampler using the edge index from the dataset.
        self.graph_sampler = NeighborSampler(
            dataset.data.edge_index,
            node_idx=None,
            sizes=layer_sizes,
            batch_size=1024,
            shuffle=True,
            num_workers=0
        )
        # If masked sampling is enabled, initialize an alternative sampler on the masked edges.
        if masked_sampler:
            self.graph_sampler_masked = NeighborSampler(
                dataset.data.edge_index[:, ~mask],
                node_idx=None,
                sizes=layer_sizes,
                batch_size=1024,
                shuffle=True,
                num_workers=0
            )
        # Create a multiprocessing queue with a capacity proportional to the number of processes.
        self.queue = Queue(num_procs * 3)
        self.num_procs = num_procs
        self.num_epochs = num_epochs
        self.completed_processes = 0  # Counter for processes that have finished sampling.
        self.layer_sizes = layer_sizes
        self.batch_cnt = 0  # Counter for the number of batches processed.
        self._start()  # Start the sampling processes or generator.
        # assert num_procs > 1

    def __len__(self):
        """
        Returns the total number of batches across all epochs.
        Raises an assertion error if num_epochs is set to -1 (indefinite sampling).
        """
        assert self.num_epochs != -1
        return self.num_epochs * self.len

    def __iter__(self):
        """
        Returns the iterator object (self).
        """
        return self

    def renew(self):
        """
        Renew the sampler for another run, ensuring that num_epochs > 0.
        
        Returns:
            A new instance of MultiProcessSampler with the same configuration.
        """
        assert self.num_epochs > 0
        return MultiProcessSampler(
            dataset=self.dataset,
            neg_sample_size=self.neg_sample_size,
            batch_size=self.batch_size,
            num_procs=self.num_procs,
            layer_sizes=self.layer_sizes,
            num_epochs=self.num_epochs,
            enforce_type=self.enforce_type,
            filter_positive=self.filter_positive
        )

    def _start(self):
        """
        Start the sampling process(es).

        If more than one process is specified, start multiple processes that call the 
        queue_producer function to generate batches. Otherwise, initialize a single-process generator.
        """
        if self.num_procs > 1:
            procs = []
            for i in range(self.num_procs):
                proc = mp.Process(
                    target=queue_producer,
                    args=(
                        self.dataset, self.neg_sample_size, self.batch_size,
                        self.enforce_type, self.filter_positive, self.num_epochs, self.queue
                    )
                )
                procs.append(proc)
                proc.start()  # Start the process.
            self.procs = procs
        else:
            self.procs = []
            # Use a single-process generator if num_procs is 1.
            self.generator = Sampler(
                self.dataset,
                self.neg_sample_size,
                self.batch_size,
                self.enforce_type,
                self.filter_positive
            ).epoch_sampler(self.num_epochs)

    def __next__(self):
        """
        Retrieve the next batch of data.

        For multi-process mode, it waits for batches from the multiprocessing queue.
        For single-process mode, it gets the next batch from the generator.
        
        Returns:
            A tuple (triples, nodes, sampled_neighbors) where:
                - triples: A list containing positive and negative triples.
                - nodes: The nodes involved in the batch.
                - sampled_neighbors: The result of neighbor sampling on the nodes.
        """
        if self.num_procs > 1:
            batch = None
            # Loop until a valid batch is retrieved or all processes have completed.
            while batch is None and self.completed_processes < self.num_procs:
                batch = self.queue.get()
                if batch is None:
                    # Increment count when a process signals completion.
                    self.completed_processes += 1
            if batch is None:
                raise StopIteration
        else:
            # Single-process mode: get the next batch from the generator.
            batch = next(self.generator)
        triples, nodes = batch
        self.batch_cnt += 1
        graph_sampler = self.graph_sampler
        # Optionally use the masked sampler based on the mask_ratio.
        if self.masked_sampler and self.batch_cnt % self.mask_ratio == 0:
            graph_sampler = self.graph_sampler_masked
        # Return the batch data along with the neighbor sampling result.
        return triples, nodes, graph_sampler.sample(nodes)

    def __del__(self):
        """
        Destructor to clean up and terminate all spawned processes.
        """
        for proc in self.procs:
            proc.terminate()
        print("Closed all processes")


class Sampler:
    """
    A basic sampler that produces batches of positive triples along with negative samples.
    
    This class handles negative sampling by corrupting the head and tail nodes of each triple.
    """
    def __init__(self, dataset, neg_sample_size, batch_size, enforce_type=False, filter_positive=False):
        self.dataset = dataset
        self.neg_sample_size = neg_sample_size
        # Ensure negative sample size is even.
        assert neg_sample_size % 2 == 0
        self.batch_size = batch_size
        # Compute the number of batches per epoch.
        self.len = len(dataset.triples[0]) // batch_size
        self.filter_positive = filter_positive
        self.enforce_type = enforce_type

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return self.len

    def epoch_sampler(self, num_epochs):
        """
        Generator that yields batches for a specified number of epochs.
        
        Args:
            num_epochs (int): The number of epochs to iterate over. If -1, iterates indefinitely.
        
        Yields:
            Batches produced by the sample() method.
        """
        epoch = 0
        while num_epochs == -1 or epoch < num_epochs:
            for batch in self.sample():
                yield batch
            epoch += 1

    def sample(self):
        """
        Generate batches of data with negative sampling.
        
        For each batch, this method:
          - Randomly permutes the indices of the positive triples.
          - Selects a batch and extracts the head (h), relation (r), and tail (t) components.
          - Generates negative samples by corrupting head and tail nodes.
          - Uses numpy.unique to map node indices to a unique set for efficient neighbor sampling.
        
        Yields:
            A tuple containing:
                - A list with positive triple indices and negative sample indices for the tail.
                - Negative sample indices reshaped for heads.
                - Negative sample indices reshaped for tails.
              and the unique set of nodes.
        """
        h, r, t = self.dataset.triples
        num_pos_triples = len(h)

        # Randomly permute the indices of the positive triples.
        perm = np.random.permutation(num_pos_triples)
        iter_size = self.batch_size
        for i in range(0, num_pos_triples, iter_size):
            batch = perm[i:i + iter_size]
            # Extract the batch triples.
            h_i, r_i, t_i = h[batch], r[batch], t[batch]

            num_edges = len(batch)
            # Compute the number of corruptions for head and tail.
            num_corruption = self.neg_sample_size // 2
            
            # TODO: OGB samples from only types of nodes that be connetcec vua edge
            # maybe beneficial for relations where both sides are less in number
            # TODO: there is some subsampling weight (read more about it)
            
            # Generate corrupted head nodes.
            corrupted_heads = np.random.randint(
                0, self.dataset.data.num_nodes, num_edges * num_corruption
            )

            # Generate corrupted tail nodes.
            corrupted_tails = np.random.randint(
                0, self.dataset.data.num_nodes, num_edges * num_corruption
            )
            # Combine positive and negative nodes and get unique nodes and inverse indices.
            nodes, node_idx = np.unique(
                np.concatenate([corrupted_heads, corrupted_tails, h_i, t_i]),
                return_inverse=True
            )
            # Yield the constructed batch:
            #   - The first element is a list containing positive head indices, relations, and positive tail indices.
            #   - The next two elements are the reshaped corrupted head and tail indices.
            #   - Also yield the unique nodes used.
            yield (
                [
                    node_idx[
                        len(corrupted_heads) + len(corrupted_tails):
                        len(corrupted_heads) + len(corrupted_tails) + len(h_i)
                    ],
                    r_i,
                    node_idx[
                        len(corrupted_heads) + len(corrupted_tails) + len(h_i):
                    ]
                ],
                node_idx[:len(corrupted_heads)].reshape(num_edges, num_corruption),
                node_idx[len(corrupted_heads):len(corrupted_heads) + len(corrupted_tails)].reshape(num_edges, num_corruption)
            ), nodes
            # print("here4")


def queue_producer(dataset, neg_sample_size, batch_size, enforce_type,
                   filter_positive, num_epochs, queue: Queue):
    """
    A producer function to be run in a separate process.

    This function creates a Sampler instance and puts batches onto the provided queue.
    Once sampling is complete for all epochs, it puts a None value to signal completion.

    Args:
        dataset: The dataset used for sampling.
        neg_sample_size (int): The number of negative samples to generate.
        batch_size (int): The batch size for sampling.
        enforce_type (bool): Whether to enforce type constraints during sampling.
        filter_positive (bool): Whether to filter out positive edges.
        num_epochs (int): The number of epochs to sample (-1 for indefinite).
        queue (Queue): The multiprocessing queue to put the batches into.
    """
    try:
        sampler = Sampler(dataset, neg_sample_size, batch_size, enforce_type, filter_positive)
        for batch in sampler.epoch_sampler(num_epochs):
            queue.put(batch)
        # Signal that this producer is finished by putting None in the queue.
        queue.put(None)
    except Exception as e:
        print("error: ", e)

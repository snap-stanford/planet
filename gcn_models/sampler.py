import multiprocessing as mp
from multiprocessing import Queue

import numpy as np
from torch_geometric.data import NeighborSampler


# NOTE: Not filtering positive edges no
class MultiProcessSampler:
    def __init__(self, *, dataset, neg_sample_size, batch_size, num_procs, layer_sizes, num_epochs=-1,
                 masked_sampler: bool = False, mask=None, mask_ratio=0,
                 enforce_type=False,
                 filter_positive=False):
        self.dataset = dataset
        self.neg_sample_size = neg_sample_size
        assert neg_sample_size % 2 == 0
        self.batch_size = batch_size
        self.len = len(dataset.triples[0]) // batch_size
        self.filter_positive = filter_positive
        self.enforce_type = enforce_type
        self.masked_sampler = masked_sampler
        self.mask_ratio = mask_ratio
        self.graph_sampler = NeighborSampler(dataset.data.edge_index, node_idx=None,
                                             sizes=layer_sizes, batch_size=1024,
                                             shuffle=True, num_workers=0)
        if masked_sampler:
            self.graph_sampler_masked = NeighborSampler(dataset.data.edge_index[:, ~mask], node_idx=None,
                                                        sizes=layer_sizes, batch_size=1024,
                                                        shuffle=True, num_workers=0)
        self.queue = Queue(num_procs * 3)
        self.num_procs = num_procs
        self.num_epochs = num_epochs
        self.completed_processes = 0
        self.layer_sizes = layer_sizes
        self.batch_cnt = 0
        self._start()
        # assert num_procs > 1

    def __len__(self):
        assert self.num_epochs != -1
        return self.num_epochs * self.len

    def __iter__(self):
        return self

    def renew(self):
        assert self.num_epochs > 0
        return MultiProcessSampler(dataset=self.dataset, neg_sample_size=self.neg_sample_size,
                                   batch_size=self.batch_size, num_procs=self.num_procs,
                                   layer_sizes=self.layer_sizes,
                                   num_epochs=self.num_epochs,
                                   enforce_type=self.enforce_type,
                                   filter_positive=self.filter_positive)

    def _start(self):
        if self.num_procs > 1:
            procs = []
            for i in range(self.num_procs):
                proc = mp.Process(target=queue_producer,
                                  args=(self.dataset, self.neg_sample_size, self.batch_size, self.enforce_type,
                                        self.filter_positive, self.num_epochs, self.queue))
                procs.append(proc)
                proc.start()
            self.procs = procs
        else:
            self.procs = []
            self.generator = Sampler(self.dataset, self.neg_sample_size, self.batch_size, self.enforce_type,
                                     self.filter_positive).epoch_sampler(self.num_epochs)

    def __next__(self):
        if self.num_procs > 1:
            batch = None
            while batch is None and self.completed_processes < self.num_procs:
                batch = self.queue.get()
                if batch is None:
                    self.completed_processes += 1
            if batch is None:
                raise StopIteration
        else:
            batch = next(self.generator)
        triples, nodes = batch
        self.batch_cnt += 1
        graph_sampler = self.graph_sampler
        if self.masked_sampler and self.batch_cnt % self.mask_ratio == 0:
            graph_sampler = self.graph_sampler_masked
        return triples, nodes, graph_sampler.sample(nodes)

    def __del__(self):
        for proc in self.procs:
            proc.terminate()
        print("Closed all processes")


class Sampler:
    def __init__(self, dataset, neg_sample_size, batch_size, enforce_type=False, filter_positive=False):
        self.dataset = dataset
        self.neg_sample_size = neg_sample_size
        assert neg_sample_size % 2 == 0
        self.batch_size = batch_size
        self.len = len(dataset.triples[0]) // batch_size
        self.filter_positive = filter_positive
        self.enforce_type = enforce_type

    def __len__(self):
        return self.len

    def epoch_sampler(self, num_epochs):
        epoch = 0
        while num_epochs == -1 or epoch < num_epochs:
            for batch in self.sample():
                yield batch
            epoch += 1

    def sample(self):
        h, r, t = self.dataset.triples
        num_pos_triples = len(h)

        perm = np.random.permutation(num_pos_triples)
        iter_size = self.batch_size
        for i in range(0, num_pos_triples, iter_size):
            batch = perm[i:i + iter_size]
            h_i, r_i, t_i = h[batch], r[batch], t[batch]

            num_edges = len(batch)
            num_corruption = self.neg_sample_size // 2
            # TODO: OGB samples from only types of nodes that be connetcec vua edge
            # maybe beneficial for relations where both sides are less in number
            # TODO: there is some subsampling weight (read more about it)
            # corrupt head
            corrupted_heads = np.random.randint(0, self.dataset.data.num_nodes, num_edges * num_corruption)

            # corrupt tail
            corrupted_tails = np.random.randint(0, self.dataset.data.num_nodes, num_edges * num_corruption)
            nodes, node_idx = np.unique(np.concatenate([corrupted_heads, corrupted_tails, h_i, t_i]),
                                        return_inverse=True)
            # print("here3")
            yield ([
                       node_idx[
                       len(corrupted_heads) + len(corrupted_tails): len(corrupted_heads) + len(corrupted_tails) + len(
                           h_i)],
                       r_i, node_idx[len(corrupted_heads) + len(corrupted_tails) + len(h_i):]
                   ],
                   node_idx[:len(corrupted_heads)].reshape(num_edges, num_corruption),
                   node_idx[len(corrupted_heads):len(corrupted_heads) + len(corrupted_tails)].reshape(num_edges,
                                                                                                      num_corruption)
                  ), nodes  # , self.graph_sampler.sample(nodes)
            # print("here4")


def queue_producer(dataset, neg_sample_size, batch_size, enforce_type,
                   filter_positive, num_epochs, queue: Queue):
    try:
        sampler = Sampler(dataset, neg_sample_size, batch_size, enforce_type,
                          filter_positive)
        for batch in sampler.epoch_sampler(num_epochs):
            queue.put(batch)
        queue.put(None)
    except Exception as e:
        print("error: ", e)

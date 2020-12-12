"""
ann-benchmarks interface for Apache Lucene.
"""

import sklearn.preprocessing
from struct import Struct
import subprocess
import sys

from ann_benchmarks.algorithms.base import BaseANN

class LuceneBatch(BaseANN):
    """
    KNN using the Lucene Vector datatype.
    """

    def __init__(self, metric: str, dimension: int, param):
        self.name = f"luceneknn dim={dimension} {param}"
        self.metric = metric
        self.dimension = dimension
        self.param = param
        self.short_name = f"luceneknn-{dimension}-{param['M']}-{param['efConstruction']}"
        self.n_iters = -1
        self.train_size = -1
        #if self.metric not in ("euclidean", "angular"):
        if self.metric != "angular":
            raise NotImplementedError(f"Not implemented for metric {self.metric}")

    def fit(self, X):
        # X is a numpy array
        if self.dimension != X.shape[1]:
            raise ArgumentError(f"Configured dimension {self.dimension} but data has shape {X.shape}")
        if self.metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        self.train_size = X.shape[0]
        filename = self.short_name + ".train"
        with open(filename, "wb") as f:
            X.tofile(f, format='f32')
        self.knn_tester('-ndoc', str(X.shape[0]),
                        '-reindex',
                        '-docs', filename,
                        '-maxConn', str(self.param['M']),
                        '-beamWidthIndex', str(self.param['efConstruction']))

    def set_query_arguments(self, fanout):
        self.fanout = fanout

    def query(self, q, n):
        raise NotImplementedError(f"Single query testing not implemented: use -batch mode only")

    def prepare_batch_query(self, X, n):
        if self.metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        data_filename = self.short_name + ".test"
        with open(data_filename, "w") as f:
            X.tofile(f)
        self.n_iters = X.shape[0]
        self.topK = n
        
    def run_batch_query(self):
        data_filename = self.short_name + ".test"
        output_filename = self.short_name + ".out"
        self.knn_tester('-niter', str(self.n_iters), '-topK', str(self.topK),
                        '-fanout', str(self.fanout),
                        '-search', data_filename,
                        '-warm', '0',
                        '-docs', self.short_name + '.train',
                        '-out', output_filename)

    def get_batch_results(self):
        output_filename = self.short_name + ".out"
        batch_res = []
        with open(output_filename, 'rb') as results:
            fmt = Struct('i' * self.topK)
            for i in range(self.n_iters):
                res = fmt.unpack(results.read(self.topK * 4))
                if max(res) >= self.train_size:
                    raise AssertionError("{res} >= {self.train_size} at {i}, results={res[0]},{res[1]},{res[2]}...")
                batch_res.append(res)
        assert len(batch_res) == self.n_iters
        return batch_res

    def knn_tester(self, *args):
        cmd = ['java',
               '-cp', 'lib/*:classes',
               '-Xmx2g', '-Xms2g',
               'org.apache.lucene.util.hnsw.KnnGraphTester',
               '-dim', str(self.dimension)
        ] + list(args)
        sys.stderr.write(str(cmd))
        subprocess.run(cmd)


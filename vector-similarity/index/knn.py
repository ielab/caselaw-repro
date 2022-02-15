import nmslib
import numpy as np
import intervaltree
import pickle
import annoy

class AnnoyKnnIndex:

    def __init__(self, dim=100):
        self.ind = annoy.AnnoyIndex(dim, 'angular')
        self.id_lookup = intervaltree.IntervalTree()
        self.lookup_pos = {}
        self.num_items = 0

    def add(self, vec: np.array, _id: str):
        start = self.num_items
        for v in vec:
            self.ind.add_item(self.num_items, v)
            self.num_items += 1

        self.id_lookup.addi(start, self.num_items, _id)
        self.lookup_pos[_id] = (start, self.num_items)

    def create_index(self):
        self.ind.build(100)

    def save(self, path: str):
        self.ind.save(path)

        with open(path+'.ids', mode='wb+') as f:
            pickle.dump(self.id_lookup, f)

        with open(path + '.lp', mode='wb+') as f:
            pickle.dump(self.lookup_pos, f)

    def load(self, path: str):
        self.ind.load(path)

        with open(path+'.ids', 'rb') as f:
                self.id_lookup = pickle.load(f)

        with open(path + '.lp', 'rb') as f:
                self.lookup_pos = pickle.load(f)

    def search(self, vec, k=100, data=None):
        rows, distances = self.ind.get_nns_by_vector(vec, k, search_k=-1, include_distances=True)
        ids = []
        doc_rows = []
        for row in rows:
            v = self.id_lookup.at(row).pop()
            doc_rows.append(row-v[0])
            ids.append(v[2])
        return ids, doc_rows, distances

    def get_doc_rows(self, _id):
        """
        Returns all vectors for matching doc id in index, else none.
        :param _id:
        :return:
        """
        if _id in self.lookup_pos:
                pos = self.lookup_pos[_id]
                rows = []
                for i in range(pos[0], pos[1]): 
                    rows.append(self.ind.get_item_vector(i))
                return rows

        return None


    def get_max_sim_rows(self, vec, rows):
        max_val = 0.0
        ind = 0
        for i, row in enumerate(rows):
            dist = np.linalg.norm(vec-row)
            if dist > max_val:
                max_val = dist
                ind = i

        return max_val, ind
            

class KnnIndex:
        def __init__(self):
                self.ind = nmslib.init(method='hnsw', space='cosinesimil', data_type=nmslib.DataType.DENSE_VECTOR)
                self.id_lookup = intervaltree.IntervalTree()
                # store id -> (start, end) row pos
                self.lookup_pos = {}
                self.num_items = 0

        def add(self, vec: np.array, _id: str):
                """
                For some reason the inserting of ids for each data point resulted in an error. As list of rows, store start
                and end row for current item to be used in range tree lookup to return id.
                """
                n = self.ind.addDataPointBatch(vec)
                to = self.num_items + n
                self.id_lookup.addi(self.num_items, to, _id)
                self.lookup_pos[_id] = (self.num_items, to)
                self.num_items = to

        def create_index(self):
            self.ind.createIndex({'post': 0, 'indexThreadQty': 12, 'M': 64, 'efConstruction': 5000})

        def load(self, path):
                self.ind.loadIndex(path, True)

                with open(path+'.ids', 'rb') as f:
                        self.id_lookup = pickle.load(f)

                with open(path + '.lp', 'rb') as f:
                        self.lookup_pos = pickle.load(f)

        def save(self, path):
                self.ind.saveIndex(path, True)

                with open(path+'.ids', mode='wb+') as f:
                        pickle.dump(self.id_lookup, f)

                with open(path + '.lp', mode='wb+') as f:
                        pickle.dump(self.lookup_pos, f)

        def search(self, vec, k=100, data=None):
                rows, distances = self.ind.knnQuery(vec, k)
                print(rows, distances)
                for row in rows:
                    print(row, self.id_lookup.at(row))
                    if data is not None:
                        print(data[row], self.ind[row])
                ids = [self.id_lookup.at(x).pop()[2] for x in rows]
                return ids, rows, distances

        def get_doc_rows(self, _id):
                """
                Returns all vectors for matching doc id in index, else none.
                :param _id:
                :return:
                """
                if _id in self.lookup_pos:
                        pos = self.lookup_pos[_id]
                        return self.ind[pos[0]:pos[1]]

                return None

        def get_max_sim_rows(self, vec, rows):
            max_val = 0.0
            for row in rows:
                dist = self.getDistance(vec, self.ind.rows[row])
                if dist > max_val:
                    max_val = dist

            return max_val
            

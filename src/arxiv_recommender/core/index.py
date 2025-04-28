import logging
import numpy as np

FAISS_OK = False
try:
    import faiss # type: ignore
    FAISS_OK = True
except ImportError:
    logging.warning("Faiss not installed. ANNIndex functionality will be unavailable. Install faiss-cpu or faiss-gpu.")

class ANNIndex:
    """Wrapper for Faiss approximate nearest neighbor index."""
    def __init__(self, dim: int, factory: str = "IVF256_HNSW32", nprobe: int = 16):
        """Initializes the Faiss index.

        Args:
            dim: Dimension of the vectors.
            factory: Faiss index factory string (e.g., "IVF256,Flat", "IndexHNSWFlat").
            nprobe: Number of cells to probe for IVF indexes.
        """
        if not FAISS_OK:
            raise RuntimeError("Faiss is required but not installed.")

        self.dim = dim
        self.factory = factory
        self.nprobe = nprobe
        self.is_trained = False

        logging.info(f"Initializing Faiss index with factory='{factory}', dim={dim}, nprobe={nprobe}")
        try:
            # Handle HNSW separately as it doesn't need explicit training if Flat
            # Factory strings like "IVF256_HNSW32,PQ8" are also possible
            if "HNSW" in factory and not "IVF" in factory:
                # Simple HNSWFlat index
                self.index = faiss.IndexHNSWFlat(dim, 32) # M=32 is a common default
                self.index = faiss.IndexIDMap(self.index) # Add ID mapping
                self.is_trained = True # HNSWFlat doesn't need training
            elif "IVF" in factory:
                # IVF-based index (potentially with HNSW quantizer)
                # Example: "IVF256,Flat", "IVF1024_HNSW32,PQ8"
                # Determine the quantizer based on the factory string if complex
                # For simplicity, assume a basic Flat quantizer if not HNSW in IVF string
                if "_HNSW" in factory:
                    # e.g., IVF256_HNSW32
                    # Extract IVF list count
                    ivf_lists = int(factory.split('_')[0][3:]) # Extracts 256 from IVF256
                    quantizer = faiss.IndexHNSWFlat(dim, 32)
                    self.index = faiss.IndexIVFFlat(quantizer, dim, ivf_lists, faiss.METRIC_L2)
                else:
                    # e.g., IVF256,Flat
                    quantizer = faiss.IndexFlatL2(dim)
                    ivf_lists = int(factory.split(',')[0][3:]) # Extracts 256 from IVF256
                    self.index = faiss.IndexIVFFlat(quantizer, dim, ivf_lists, faiss.METRIC_L2)
                    # Add support for other quantizers like PQ if needed

                self.index = faiss.IndexIDMap(self.index) # Add ID mapping
                self.index.nprobe = nprobe
                self.is_trained = False # IVF needs training
            else:
                 # Default to IndexFlatL2 if factory is simple or unrecognized pattern
                 logging.warning(f"Unrecognized or simple factory '{factory}'. Defaulting to IndexFlatL2 + IDMap.")
                 base_index = faiss.IndexFlatL2(dim)
                 self.index = faiss.IndexIDMap(base_index)
                 self.is_trained = True # Flat index doesn't need training

        except Exception as e:
            logging.error(f"Failed to initialize Faiss index with factory '{factory}': {e}")
            raise

    def train(self, vecs: np.ndarray):
        """Trains the index (required for IVF types)."""
        if not self.is_trained:
            if vecs.shape[0] == 0:
                logging.warning("Skipping training index: No vectors provided.")
                return
            if vecs.shape[1] != self.dim:
                 raise ValueError(f"Training vector dimension {vecs.shape[1]} != index dimension {self.dim}")

            logging.info(f"Training index with {vecs.shape[0]} vectors...")
            try:
                self.index.train(vecs)
                self.is_trained = True
                logging.info("Index training complete.")
            except Exception as e:
                err_msg = str(e)
                # If not enough training points for IVF, fallback to Flat index
                if isinstance(e, RuntimeError) and "nx >= k" in err_msg:
                    logging.warning("Training data too small for IVF; falling back to Flat index.")
                    flat_index = faiss.IndexFlatL2(self.dim)
                    self.index = faiss.IndexIDMap(flat_index)
                    self.is_trained = True
                else:
                    logging.error(f"Error during Faiss index training: {e}")
                    raise
        else:
            logging.info("Index is already trained or does not require training.")

    def add(self, vecs: np.ndarray, ids: np.ndarray | None = None):
        """Adds vectors to the index.

        Args:
            vecs: Numpy array of vectors to add.
            ids: Optional numpy array of corresponding integer IDs.
                 If None, uses sequential IDs starting from current count.
        """
        if not self.is_trained and "IVF" in self.factory:
            raise RuntimeError("Index must be trained before adding vectors.")
        if vecs.shape[0] == 0:
            logging.warning("Skipping adding vectors: No vectors provided.")
            return
        if vecs.shape[1] != self.dim:
             raise ValueError(f"Vector dimension {vecs.shape[1]} != index dimension {self.dim}")

        if ids is not None and len(ids) != vecs.shape[0]:
            raise ValueError("Number of IDs must match number of vectors.")

        effective_ids = ids
        if isinstance(self.index, faiss.IndexIDMap):
             if ids is None:
                 # Generate sequential IDs if none provided
                 start_id = self.index.ntotal
                 effective_ids = np.arange(start_id, start_id + vecs.shape[0], dtype=np.int64)
             logging.info(f"Adding {vecs.shape[0]} vectors with IDs to IndexIDMap...")
             self.index.add_with_ids(vecs, effective_ids)
        else:
            if ids is not None:
                logging.warning("Index does not support custom IDs (not IndexIDMap). Ignoring provided IDs.")
            logging.info(f"Adding {vecs.shape[0]} vectors to index...")
            self.index.add(vecs)

        logging.info(f"Index now contains {self.index.ntotal} vectors.")

    def search(self, queries: np.ndarray, k: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Performs ANN search.

        Args:
            queries: Numpy array of query vectors.
            k: Number of nearest neighbors to retrieve.

        Returns:
            Tuple of (distances, indices).
        """
        if queries.shape[0] == 0:
            logging.warning("Skipping search: No query vectors provided.")
            # Return empty arrays matching Faiss output shape
            return np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.int64)
        if queries.shape[1] != self.dim:
             raise ValueError(f"Query vector dimension {queries.shape[1]} != index dimension {self.dim}")
        if self.index.ntotal == 0:
            logging.warning("Cannot search on an empty index.")
            return np.empty((queries.shape[0], k), dtype=np.float32), np.empty((queries.shape[0], k), dtype=np.int64)

        logging.info(f"Searching for {k} nearest neighbors for {queries.shape[0]} queries...")
        distances, indices = self.index.search(queries, k)
        # Replace -1 indices (no neighbor found) with a marker or handle as needed
        # For now, keep Faiss default (-1)
        logging.info("Search complete.")
        return distances, indices

    # Add save/load methods if needed
    # def save(self, path: str):
    #     logging.info(f"Saving index to {path}")
    #     faiss.write_index(self.index, path)

    # @classmethod
    # def load(cls, path: str):
    #     logging.info(f"Loading index from {path}")
    #     index = faiss.read_index(path)
    #     # Need to reconstruct ANNIndex object state (dim, factory, nprobe, is_trained)
    #     # This requires saving metadata alongside the index file
    #     raise NotImplementedError("Loading index requires metadata handling") 
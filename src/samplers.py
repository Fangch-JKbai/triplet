import math
import random
from typing import List, Dict, Iterator, Optional
from torch.utils.data import Sampler


class PKSampler(Sampler[List[int]]):
    """
    PK Sampler (Improved)
    - tries to avoid duplicate sample indices within a batch (important for multi-label leakage)
    - does NOT drop classes with <K samples (uses sampling with replacement when needed)
    - covers all classes each epoch (ceil), pads last batch if needed
    - reproducible via internal RNG (seed)
    NOTE: This sampler yields a LIST of indices each step -> should be used as DataLoader(batch_sampler=...)
    """

    def __init__(
        self,
        ec_to_indices: Dict[str, List[int]],
        p: int,
        k: int,
        seed: int = 42,
        drop_last: bool = False,
        max_resample: int = 50,
    ):
        if not isinstance(p, int) or p <= 0:
            raise ValueError(f"p must be a positive integer, but got {p}")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, but got {k}")

        # Keep only non-empty classes
        self.ec_to_indices = {ec: idxs for ec, idxs in ec_to_indices.items() if len(idxs) > 0}
        if not self.ec_to_indices:
            raise ValueError("ec_to_indices is empty after filtering empty classes.")

        self.p = p
        self.k = k
        self.drop_last = drop_last
        self.max_resample = int(max_resample)

        self.ecs: List[str] = list(self.ec_to_indices.keys())
        self.num_classes = len(self.ecs)

        self.rng = random.Random(seed)

        if self.drop_last:
            self.num_batches = self.num_classes // self.p
        else:
            self.num_batches = math.ceil(self.num_classes / self.p)

        if self.num_batches <= 0:
            raise ValueError("num_batches computed as 0. Check p and number of classes.")

    def __len__(self) -> int:
        return self.num_batches

    def _sample_k_from_class(self, ec: str, used: set) -> List[int]:
        """Sample K indices from a class, trying to avoid used indices (no replacement if possible)."""
        idxs = self.ec_to_indices[ec]
        out: List[int] = []

        # Prefer unused indices first
        candidates_unused = [x for x in idxs if x not in used]

        if len(candidates_unused) >= self.k:
            # sample without replacement from unused
            chosen = self.rng.sample(candidates_unused, self.k)
            out.extend(chosen)
            return out

        # take all unused first
        out.extend(candidates_unused)

        # need more to reach K
        need = self.k - len(out)

        if need <= 0:
            return out

        # If total idxs >= need, we can sample remaining from all (may include used)
        # Use replacement if class is very small
        if len(idxs) == 1:
            out.extend([idxs[0]] * need)
            return out

        # Try to pick extra indices with minimal duplication inside this class chunk
        # Allow replacement if not enough unique remain
        for _ in range(need):
            # attempt to avoid duplicates inside out
            picked = None
            for _try in range(self.max_resample):
                cand = self.rng.choice(idxs)
                if cand not in out:
                    picked = cand
                    break
            if picked is None:
                picked = self.rng.choice(idxs)  # fallback
            out.append(picked)

        return out

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle classes each epoch (reproducible with internal RNG state)
        ecs = self.ecs[:]
        self.rng.shuffle(ecs)

        # Chunk into batches of classes
        for b in range(self.num_batches):
            start = b * self.p
            end = start + self.p
            class_batch = ecs[start:end]

            if len(class_batch) < self.p:
                if self.drop_last:
                    break
                # pad with random classes (with replacement)
                pad = self.p - len(class_batch)
                class_batch = class_batch + [self.rng.choice(self.ecs) for _ in range(pad)]

            used = set()
            batch_indices: List[int] = []

            for ec in class_batch:
                chosen = self._sample_k_from_class(ec, used)

                # try to avoid duplicates across classes: resample a few times if collisions
                # (best-effort; if dataset is tiny, duplicates may still happen)
                if any(x in used for x in chosen):
                    # attempt to resample within this class to reduce collisions
                    for _ in range(self.max_resample):
                        alt = self._sample_k_from_class(ec, used)
                        if not any(x in used for x in alt):
                            chosen = alt
                            break

                batch_indices.extend(chosen)
                used.update(chosen)

            yield batch_indices

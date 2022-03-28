import equinox as eqx

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as lse

class Hmm(eqx.Module):
    start: jnp.ndarray
    trans: jnp.ndarray
    emit: jnp.ndarray

    def __init__(self, state_size, out_size, key):
        skey, tkey, ekey = jax.random.split(key, 3)

        start = jax.random.uniform(skey, (state_size,))
        trans = jax.random.uniform(tkey, (state_size, state_size))
        emit = jax.random.uniform(ekey, (state_size, out_size))

        self.start = start - lse(start, 0)
        self.trans = trans - lse(trans, -1, keepdims=True)
        self.emit = emit - lse(emit, -1, keepdims=True)

    def forward(self, xs):
        T = xs.shape[0]
        emit = self.emit[:,xs]
        evidences = []
        un_alpha = self.start + emit[:,0]
        Z = lse(un_alpha, 0, keepdims=True)
        alpha = un_alpha - Z
        evidences.append(Z)
        for t in range(1, T):
            un_alpha = lse(alpha[:,None] + self.trans + emit[None,:,t], 0)
            Z = lse(un_alpha, 0, keepdims=True)
            alpha = un_alpha - Z
            evidences.append(Z)
        return jnp.concatenate(evidences)

    def sample(self, key, T):
        import pdb; pdb.set_trace()

def main():
    Z, X, T, B = 16, 23, 7, 32
    key = jax.random.PRNGKey(1234)

    hmm = Hmm(16, 23, key=key)

    xs_key = jax.random.PRNGKey(0)
    xs = jax.random.randint(xs_key, (T,), 0, X, dtype=int)
    evidence = hmm.forward(xs)

    new_xs = hmm.sample(key, T)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()

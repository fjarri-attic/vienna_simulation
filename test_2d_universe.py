import numpy
from reikna.cluda import ocl_api, dtypes, Module, functions

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reikna.helpers import product
from integrator import Integrator


def nonlinear_no_potential(dtype, interaction, tunneling):
    r"""
    Nonlinear module

    .. math::

        N(\psi_1, ... \psi_C)
        = \sum_{n=1}^{C} U_{jn} |\psi_n|^2 \psi_j
          - \nu_j psi_{m_j}

    ``interaction``: a symmetrical ``components x components`` array with interaction strengths.
    ``tunneling``: a list of (other_comp, coeff) pairs of tunnelling strengths.
    """

    c_dtype = dtype
    c_ctype = dtypes.ctype(c_dtype)
    s_dtype = dtypes.real_for(dtype)
    s_ctype = dtypes.ctype(s_dtype)

    return Module.create(
        """
        %for comp in range(components):
        INLINE WITHIN_KERNEL ${c_ctype} ${prefix}${comp}(
            %for pcomp in range(components):
            ${c_ctype} psi${pcomp},
            %endfor
            ${s_ctype} V, ${s_ctype} t)
        {
            return (
                ${mul}(psi${comp}, (
                    %for other_comp in range(components):
                    + ${dtypes.c_constant(interaction[comp, other_comp], s_dtype)} *
                        ${norm}(psi${other_comp})
                    %endfor
                    ))
                - ${mul}(
                    psi${tunneling[comp][0]},
                    ${dtypes.c_constant(tunneling[comp][1], s_dtype)})
                );
        }
        %endfor
        """,
        render_kwds=dict(
            components=interaction.shape[0],
            mul=functions.mul(c_dtype, s_dtype),
            norm=functions.norm(c_dtype),
            interaction=interaction,
            tunneling=tunneling,
            s_dtype=s_dtype,
            c_ctype=c_ctype,
            s_ctype=s_ctype))


def beam_splitter(psi):
    psi_c = psi.copy()
    psi_c[0] = (psi[0] + psi[1]) / numpy.sqrt(2)
    psi_c[1] = (psi[0] - psi[1]) / numpy.sqrt(2)
    return psi_c


class CollectorWigner2D:

    def __init__(self, dV):
        self.dV = dV

    def __call__(self, psi):
        psi = beam_splitter(psi.get())

        ns = numpy.abs(psi) ** 2 - 0.5 / self.dV
        n = ns.mean(1)
        Ns = (ns * self.dV).sum(-1).sum(-1)

        res = dict(
            Nplus_mean=Ns[0].mean(), Nminus_mean=Ns[1].mean(),
            Nplus_std=Ns[0].std(), Nminus_std=Ns[1].std(),
            density=n)

        return res


def test_2d_universe():

    modes = 128
    L_trap = 80.
    gamma = 0.1
    N = L_trap ** 2 / gamma
    samples = 10
    t = 10.
    nu = 0.1
    steps = 5000
    ensembles = 1
    dtype = numpy.complex128

    problem_shape = (modes, modes)
    shape = (2, ensembles) + problem_shape
    box = (L_trap, L_trap)
    dV = (L_trap / modes) ** 2

    api = ocl_api()
    #device = api.get_platforms()[0].get_devices()[1]
    #thr = api.Thread(device)
    thr = api.Thread.create()

    interaction = numpy.array([[gamma, 0], [0, gamma]])
    tunneling = [(1, nu), (0, nu)]
    nonlinear_module = nonlinear_no_potential(dtype, interaction, tunneling)

    psi = numpy.empty(shape, dtype)

    integrator = Integrator(thr, psi, box, t, steps, samples,
        kinetic_coeff=0.5,
        nonlinear_module=nonlinear_module)


    # Classical ground state
    psi.fill((1. / gamma) ** 0.5)
    psi[1] *= -1 # opposite phases of components


    # To Wigner
    rs = numpy.random.RandomState(seed=123)
    normals = rs.normal(size=(2,) + shape, scale=numpy.sqrt(0.5))
    noise_kspace = numpy.sqrt(0.5) * (normals[0] + 1j * normals[1])

    fft_scale = numpy.sqrt(dV / product(problem_shape))
    psi += numpy.fft.ifftn(noise_kspace, axes=range(2, len(shape))) / fft_scale


    psi_dev = thr.to_device(psi)
    collector = CollectorWigner2D(dV)
    results = integrator(psi_dev, [collector])

    print("Errors:", results.errors)
    assert results.errors['density'] < 1e-7
    assert results.errors['psi_strong_mean'] < 1e-7
    assert results.errors['psi_strong_max'] < 1e-7

    # Check that the population stayed constant
    N_total = results.values['Nplus_mean'] + results.values['Nminus_mean']
    # Not using N, since the initial value can differ slightly (due to initial sampling)
    N_diff = (N_total - N_total[0]) / N_total[0]
    assert numpy.abs(N_diff).max() < 1e-6

    plot_2d_universe(results.values['density'][-1], L_trap, N)


def plot_2d_universe(n, L, N):

    levels = numpy.linspace(-0.5, 0.5, 11)

    fig = plt.figure()
    s = fig.add_subplot(111, xlabel='$x$', ylabel='$y$')

    n_max = 4 * N / (L**2)
    data = (n[0] - n[1]).T / n_max
    data = numpy.clip(data, -0.5, 0.5)

    im = s.imshow(data,
        extent=(-L/2, L/2, -L/2, L/2),
        vmin=-0.5, vmax=0.5,
        interpolation='none',
        aspect=1)

    fig.colorbar(im,orientation='vertical', shrink=0.6, ticks=levels).set_label('$j$')
    fig.tight_layout()
    fig.savefig('test_2d_universe.pdf')


if __name__ == '__main__':
    test_2d_universe()

import numpy
from reikna.cluda import ocl_api, dtypes, Module, functions

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reikna.helpers import product
from integrator import Integrator


def get_nonlinear(dtype, interaction, tunneling):
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
                    + V
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



class CollectorWigner1D:

    def __init__(self, dV):
        self.dV = dV

    def __call__(self, psi):
        psi = psi.get()

        ns = numpy.abs(psi) ** 2 - 0.5 / self.dV
        n = ns.mean(1)
        Ns = (ns * self.dV).sum(-1)

        res = dict(
            N=Ns[0].mean(),
            N_std=Ns[0].std(),
            density=n[0])

        return res


def test_soliton():

    seed = 31415926 # random seed
    modes = 128 # spatial lattice points
    L_trap = 14. # spatial domain
    ensembles = 64 # simulation paths
    gamma = 0.1
    t = 2.5 # time interval
    samples = 100 # how many samples to take during simulation
    steps = samples * 400 # number of time steps (should be multiple of samples)
    v = 40.0 # strength of the potential
    soliton_height = 10.0
    soliton_shift = 1.0
    dtype = numpy.complex128

    problem_shape = (modes,)
    shape = (1, ensembles) + problem_shape
    box = (L_trap,)
    dV = L_trap / modes
    xgrid = numpy.linspace(-L_trap/2 + dV/2, L_trap/2 - dV/2, modes)

    api = ocl_api()
    #device = api.get_platforms()[0].get_devices()[1]
    #thr = api.Thread(device)
    thr = api.Thread.create()

    interaction = numpy.array([[gamma]])
    tunneling = [(0, 0)]
    nonlinear_module = get_nonlinear(dtype, interaction, tunneling)
    potential = v * xgrid ** 2

    psi = numpy.empty(shape, dtype)

    integrator = Integrator(thr, psi, box, t, steps, samples,
        kinetic_coeff=0.5,
        nonlinear_module=nonlinear_module,
        potentials=potential)


    # Classical ground state
    psi = soliton_height / numpy.cosh(xgrid - soliton_shift)
    psi = psi.reshape(1, 1, *psi.shape).astype(dtype)
    psi = numpy.tile(psi, (1, ensembles, 1))

    # To Wigner
    rs = numpy.random.RandomState(seed=456)
    normals = rs.normal(size=(2,) + shape, scale=numpy.sqrt(0.5))
    noise_kspace = numpy.sqrt(0.5) * (normals[0] + 1j * normals[1])

    fft_scale = numpy.sqrt(dV / product(problem_shape))
    psi += numpy.fft.ifftn(noise_kspace, axes=range(2, len(shape))) / fft_scale

    psi_dev = thr.to_device(psi)
    collector = CollectorWigner1D(dV)
    results = integrator(psi_dev, [collector])

    print("Errors:", results.errors)
    # TODO: what causes the errors this big? there seems to be plenty of time steps
    assert results.errors['density'] < 1e-4
    assert results.errors['psi_strong_mean'] < 0.01
    assert results.errors['psi_strong_max'] < 0.01

    # Check that the population stayed constant
    N_total = results.values['N']
    # Not using N, since the initial value can differ slightly (due to initial sampling)
    N_diff = (N_total - N_total[0]) / N_total[0]
    assert numpy.abs(N_diff).max() < 1e-5

    plot_soliton(results.values['density'], L_trap, soliton_height ** 2, t)


def plot_soliton(n, L, n_max, t):

    fig = plt.figure()
    s = fig.add_subplot(111, xlabel='$t$', ylabel='$x$')

    im = s.imshow(n.T,
        extent=(0, t, -L/2, L/2),
        vmin=0, vmax=n_max,
        interpolation='none',
        aspect='auto')

    fig.colorbar(im,orientation='vertical', shrink=0.6).set_label('$n$')
    fig.tight_layout()
    fig.savefig('test_soliton.pdf')


if __name__ == '__main__':
    test_soliton()

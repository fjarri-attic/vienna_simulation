from __future__ import print_function

import numpy
import sys
import time

from reikna.helpers import product
from reikna.cluda import dtypes, Module, Snippet, functions
from reikna.core import Computation, Parameter, Annotation, Transformation, Type

from reikna.fft import FFT
from reikna.pureparallel import PureParallel

_range = xrange if sys.version_info[0] < 3 else range


def get_ksquared(shape, box):
    ks = [
        2 * numpy.pi * numpy.fft.fftfreq(size, length / size)
        for size, length in zip(shape, box)]

    if len(shape) > 1:
        full_ks = numpy.meshgrid(*ks, indexing='ij')
    else:
        full_ks = ks

    return sum([full_k ** 2 for full_k in full_ks])


def get_nonlinear_wrapper(components, c_dtype, nonlinear_module, dt):
    s_dtype = dtypes.real_for(c_dtype)
    return Module.create(
        """
        %for comp in range(components):
        INLINE WITHIN_KERNEL ${c_ctype} ${prefix}${comp}(
            %for pcomp in range(components):
            ${c_ctype} psi${pcomp},
            %endfor
            ${s_ctype} V, ${s_ctype} t)
        {
            ${c_ctype} nonlinear = ${nonlinear}${comp}(
                %for pcomp in range(components):
                psi${pcomp},
                %endfor
                V, t);
            return ${mul}(
                COMPLEX_CTR(${c_ctype})(0, -${dt}),
                nonlinear);
        }
        %endfor
        """,
        render_kwds=dict(
            components=components,
            c_ctype=dtypes.ctype(c_dtype),
            s_ctype=dtypes.ctype(s_dtype),
            mul=functions.mul(c_dtype, c_dtype),
            dt=dtypes.c_constant(dt, s_dtype),
            nonlinear=nonlinear_module))


def get_nonlinear1(state_arr, potential_arr, scalar_dtype, nonlinear_module):
    # output = N(input)
    return PureParallel(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i')),
            Parameter('potential_curr', Annotation(potential_arr, 'i')),
            Parameter('t', Annotation(scalar_dtype))],
        """
        %for comp in range(components):
        ${output.ctype} psi${comp} = ${input.load_idx}(${comp}, ${idxs.all()});
        %endfor
        ${potential_curr.ctype} V = ${potential_curr.load_idx}(${', '.join(idxs[1:])});

        %for comp in range(components):
        ${output.store_idx}(${comp}, ${idxs.all()}, ${nonlinear}${comp}(
            %for pcomp in range(components):
            psi${pcomp},
            %endfor
            V, ${t}));
        %endfor
        """,
        guiding_array=state_arr.shape[1:],
        render_kwds=dict(components=state_arr.shape[0], nonlinear=nonlinear_module))


def get_nonlinear2(state_arr, potential_arr, scalar_dtype, nonlinear_module, dt):
    # k2 = N(psi_I + k1 / 2, t + dt / 2)
    # k3 = N(psi_I + k2 / 2, t + dt / 2)
    # psi_4 = psi_I + k3 (argument for the 4-th step k-propagation)
    # psi_k = psi_I + (k1 + 2(k2 + k3)) / 6 (argument for the final k-propagation)
    return PureParallel(
        [
            Parameter('psi_k', Annotation(state_arr, 'o')),
            Parameter('psi_4', Annotation(state_arr, 'o')),
            Parameter('psi_I', Annotation(state_arr, 'i')),
            Parameter('k1', Annotation(state_arr, 'i')),
            Parameter('potential_half', Annotation(potential_arr, 'i')),
            Parameter('t', Annotation(scalar_dtype))],
        """
        %for comp in range(components):
        ${psi_k.ctype} psi_I_${comp} = ${psi_I.load_idx}(${comp}, ${idxs.all()});
        ${psi_k.ctype} k1_${comp} = ${k1.load_idx}(${comp}, ${idxs.all()});
        %endfor
        ${potential_half.ctype} V = ${potential_half.load_idx}(${', '.join(idxs[1:])});

        %for comp in range(components):
        ${psi_k.ctype} k2_${comp} = ${nonlinear}${comp}(
            %for pcomp in range(components):
            psi_I_${pcomp} + ${div}(k1_${pcomp}, 2),
            %endfor
            V, ${t} + ${dt} / 2);
        %endfor

        %for comp in range(components):
        ${psi_k.ctype} k3_${comp} = ${nonlinear}${comp}(
            %for pcomp in range(components):
            psi_I_${pcomp} + ${div}(k2_${pcomp}, 2),
            %endfor
            V, ${t} + ${dt} / 2);
        %endfor

        %for comp in range(components):
        ${psi_4.store_idx}(${comp}, ${idxs.all()}, psi_I_${comp} + k3_${comp});
        ${psi_k.store_idx}(
            ${comp}, ${idxs.all()},
            psi_I_${comp} + ${div}(k1_${comp}, 6) + ${div}(k2_${comp}, 3) + ${div}(k3_${comp}, 3));
        %endfor
        """,
        guiding_array=state_arr.shape[1:],
        render_kwds=dict(
            components=state_arr.shape[0],
            nonlinear=nonlinear_module,
            dt=dtypes.c_constant(dt, scalar_dtype),
            div=functions.div(state_arr.dtype, numpy.int32, out_dtype=state_arr.dtype)))


def get_nonlinear3(state_arr, potential_arr, scalar_dtype, nonlinear_module, dt):
    # k4 = N(D(psi_4), t + dt)
    # output = D(psi_k) + k4 / 6
    return PureParallel(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('kprop_psi_k', Annotation(state_arr, 'i')),
            Parameter('kprop_psi_4', Annotation(state_arr, 'i')),
            Parameter('potential_next', Annotation(potential_arr, 'i')),
            Parameter('t', Annotation(scalar_dtype))],
        """
        %for comp in range(components):
        ${output.ctype} psi4_${comp} = ${kprop_psi_4.load_idx}(${comp}, ${idxs.all()});
        ${output.ctype} psik_${comp} = ${kprop_psi_k.load_idx}(${comp}, ${idxs.all()});
        %endfor
        ${potential_next.ctype} V = ${potential_next.load_idx}(${', '.join(idxs[1:])});

        %for comp in range(components):
        ${output.ctype} k4_${comp} = ${nonlinear}${comp}(
            %for pcomp in range(components):
            psi4_${pcomp},
            %endfor
            V, ${t} + ${dt});
        ${output.store_idx}(${comp}, ${idxs.all()}, psik_${comp} + ${div}(k4_${comp}, 6));
        %endfor
        """,
        guiding_array=state_arr.shape[1:],
        render_kwds=dict(
            components=state_arr.shape[0],
            nonlinear=nonlinear_module,
            dt=dtypes.c_constant(dt, scalar_dtype),
            div=functions.div(state_arr.dtype, numpy.int32, out_dtype=state_arr.dtype)))


def get_potential_interpolator(potential_arr, dt):
    scalar_dtype = potential_arr.dtype
    return PureParallel(
        [
            Parameter('potential_curr', Annotation(potential_arr, 'o')),
            Parameter('potential_half', Annotation(potential_arr, 'o')),
            Parameter('potential_next', Annotation(potential_arr, 'o')),
            Parameter('potential1', Annotation(potential_arr, 'i')),
            Parameter('potential2', Annotation(potential_arr, 'i')),
            Parameter('t_potential1', Annotation(scalar_dtype)),
            Parameter('t_potential2', Annotation(scalar_dtype)),
            Parameter('t', Annotation(scalar_dtype))],
        """
        ${potential1.ctype} p1 = ${potential1.load_idx}(${idxs.all()});
        ${potential2.ctype} p2 = ${potential2.load_idx}(${idxs.all()});

        ${t.ctype} normalization = (p2 - p1) / (${t_potential2} - ${t_potential1});
        ${t.ctype} offset = ${t} - ${t_potential1};
        ${potential_curr.store_idx}(${idxs.all()}, p1 + offset * normalization);
        ${potential_half.store_idx}(${idxs.all()}, p1 + (offset + ${dt} / 2) * normalization);
        ${potential_next.store_idx}(${idxs.all()}, p1 + (offset + ${dt}) * normalization);
        """,
        render_kwds=dict(dt=dtypes.c_constant(dt, scalar_dtype)))


class RK4IPStepper(Computation):
    """
    The integration method is RK4IP taken from the thesis by B. Caradoc-Davies
    "Vortex Dynamics in Bose-Einstein Condensates" (2000),
    namely Eqns. B.10 (p. 166).
    """

    def __init__(self, state_arr, dt, box=None, kinetic_coeff=1, nonlinear_module=None):
        scalar_dtype = dtypes.real_for(state_arr.dtype)
        potential_arr = Type(scalar_dtype, shape=state_arr.shape[2:])

        Computation.__init__(self, [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i')),
            Parameter('potential1', Annotation(potential_arr, 'i')),
            Parameter('potential2', Annotation(potential_arr, 'i')),
            Parameter('t_potential1', Annotation(scalar_dtype)),
            Parameter('t_potential2', Annotation(scalar_dtype)),
            Parameter('t', Annotation(scalar_dtype))])

        self._box = box
        self._kinetic_coeff = kinetic_coeff
        self._nonlinear_module = nonlinear_module
        self._components = state_arr.shape[0]
        self._ensembles = state_arr.shape[1]
        self._grid_shape = state_arr.shape[2:]

        ksquared = get_ksquared(self._grid_shape, self._box)
        self._kprop = numpy.exp(ksquared * (-1j * kinetic_coeff * dt / 2)).astype(state_arr.dtype)
        self._kprop_trf = Transformation(
            [
                Parameter('output', Annotation(state_arr, 'o')),
                Parameter('input', Annotation(state_arr, 'i')),
                Parameter('kprop', Annotation(self._kprop, 'i'))],
            """
            ${kprop.ctype} kprop_coeff = ${kprop.load_idx}(${', '.join(idxs[2:])});
            ${output.store_same}(${mul}(${input.load_same}, kprop_coeff));
            """,
            render_kwds=dict(mul=functions.mul(state_arr.dtype, self._kprop.dtype)))

        self._fft = FFT(state_arr, axes=range(2, len(state_arr.shape)))
        self._fft_with_kprop = FFT(state_arr, axes=range(2, len(state_arr.shape)))
        self._fft_with_kprop.parameter.output.connect(
            self._kprop_trf, self._kprop_trf.input,
            output_prime=self._kprop_trf.output,
            kprop=self._kprop_trf.kprop)

        nonlinear_wrapper = get_nonlinear_wrapper(
            state_arr.shape[0], state_arr.dtype, nonlinear_module, dt)
        self._N1 = get_nonlinear1(state_arr, potential_arr, scalar_dtype, nonlinear_wrapper)
        self._N2 = get_nonlinear2(state_arr, potential_arr, scalar_dtype, nonlinear_wrapper, dt)
        self._N3 = get_nonlinear3(state_arr, potential_arr, scalar_dtype, nonlinear_wrapper, dt)
        self._potential_interpolator = get_potential_interpolator(potential_arr, dt)

    def _add_kprop(self, plan, output, input_, kprop_device):
        temp = plan.temp_array_like(output)
        plan.computation_call(self._fft_with_kprop, temp, kprop_device, input_)
        plan.computation_call(self._fft, output, temp, inverse=True)

    def _build_plan(self, plan_factory, device_params,
            output, input_, potential1, potential2, t_potential1, t_potential2, t):

        plan = plan_factory()

        kprop_device = plan.persistent_array(self._kprop)

        # Interpoplate potentials
        potential_curr = plan.temp_array_like(potential1)
        potential_half = plan.temp_array_like(potential1)
        potential_next = plan.temp_array_like(potential1)
        plan.computation_call(
            self._potential_interpolator,
            potential_curr, potential_half, potential_next,
            potential1, potential2, t_potential1, t_potential2, t)

        # psi_I = D(psi)
        psi_I = plan.temp_array_like(output)
        self._add_kprop(plan, psi_I, input_, kprop_device)

        # k1 = D(N(psi, t))
        k1 = plan.temp_array_like(output)
        temp = plan.temp_array_like(output)
        plan.computation_call(self._N1, temp, input_, potential_curr, t)
        self._add_kprop(plan, k1, temp, kprop_device)

        # k2 = N(psi_I + k1 / 2, t + dt / 2)
        # k3 = N(psi_I + k2 / 2, t + dt / 2)
        # psi_4 = psi_I + k3 (argument for the 4-th step k-propagation)
        # psi_k = psi_I + (k1 + 2(k2 + k3)) / 6 (argument for the final k-propagation)
        psi_4 = plan.temp_array_like(output)
        psi_k = plan.temp_array_like(output)
        plan.computation_call(self._N2, psi_k, psi_4, psi_I, k1, potential_half, t)

        # k4 = N(D(psi_4), t + dt)
        # output = D(psi_k) + k4 / 6
        kprop_psi_k = plan.temp_array_like(output)
        self._add_kprop(plan, kprop_psi_k, psi_k, kprop_device)
        kprop_psi_4 = plan.temp_array_like(output)
        self._add_kprop(plan, kprop_psi_4, psi_4, kprop_device)
        plan.computation_call(self._N3, output, kprop_psi_k, kprop_psi_4, potential_next, t)

        return plan


class Integrator:

    def __init__(self, thr, psi_arr_t, box, tmax, steps, samples,
            kinetic_coeff=1, nonlinear_module=None, potentials=None):
        r"""
        Integrates the equation

        .. math::

            i \frac{\psi_j}{dt} = - k \nabla^2 psi_j + N(\psi_1, ... \psi_C, V(t), t),

        where :math:`C` is the number of components, :math:`V` is the dynamic potential.

        ``psi_arr_t`` is an array-like object with the shape ``(components, ensembles, *grid)``.
        ``box`` is a tuple of length ``grid``, containing sizes of the simulation box.
        ``tmax`` is the propagation time.
        ``steps`` is the number of time steps to take.
        ``samples`` is the number of samples to take (not counting the initial one);
            should be a factor of ``steps``.
        ``kinetic_coeff`` is the value of :math:`k`.
        ``nonlinear_module`` calculates :math:`N`.
        ``potentials``: ``None``, an array of shape ``grid``, or an array of shape ``(M, *grid)``,
            corresponding to the values of dynamic potential.
            The dynamic potential contains the snapshots at time points from 0 to ``tmax``,
            and ``M - 1`` should be a factor of ``steps``.
        """

        self.tmax = tmax

        assert steps % samples == 0
        self.steps = steps
        self.samples = samples

        self.dt = float(tmax) / steps
        self.dt_half = self.dt / 2

        if potentials is None:
            self.potentials = numpy.zeros((2,) + psi_arr_t.shape[2:], psi_arr_t.dtype)
        elif len(potentials.shape) == len(psi_arr_t.shape) - 2:
            potentials = potentials.astype(dtypes.real_for(psi_arr_t.dtype))
            potentials = potentials.reshape(1, *(psi_arr_t.shape[2:]))
            self.potentials = numpy.vstack([potentials, potentials])
        else:
            assert len(potentials.shape) == len(psi_arr_t.shape) - 1
            assert steps % (potentials.shape[0] - 1) == 0
            potentials = potentials.astype(dtypes.real_for(psi_arr_t.dtype))
            self.potentials = potentials

        self.thr = thr
        self.stepper = RK4IPStepper(psi_arr_t, self.dt,
            box=box, kinetic_coeff=kinetic_coeff, nonlinear_module=nonlinear_module).compile(thr)
        self.stepper_half = RK4IPStepper(psi_arr_t, self.dt_half,
            box=box, kinetic_coeff=kinetic_coeff, nonlinear_module=nonlinear_module).compile(thr)

    def _collect(self, psi, t, collectors):
        t_start = time.time()

        res_dict = {'time': t}
        for collector in collectors:
            res_dict.update(collector(psi))

        t_collectors = time.time() - t_start
        return res_dict, t_collectors

    def _integrate(self, psi, half_step, collectors):

        results = []

        t_collectors = 0

        t_start = time.time()

        stepper = self.stepper_half if half_step else self.stepper
        dt = self.dt_half if half_step else self.dt
        half_step_modifier = 2 if half_step else 1
        steps = self.steps * half_step_modifier

        if half_step:
            print("Sampling at t =", end=' ')
        else:
            print("Skipping callbacks at t =", end=' ')

        if half_step:
            res_dict, t_col = self._collect(psi, 0, collectors)
            results.append(res_dict)
            t_collectors += t_col

        # Initialize potentials
        potential1 = self.thr.to_device(self.potentials[0])
        potential2 = self.thr.to_device(self.potentials[0])
        t_potential1 = 0 # time of the activation of the previous potential
        t_potential2 = 0 # time of the activation of the next potential
        potential_dt = self.tmax / (self.potentials.shape[0] - 1)

        for step in _range(steps):

            p_index, p_remainder = divmod(step, (steps // (self.potentials.shape[0] - 1)))
            if p_remainder == 0:
                potential1, potential2 = potential2, potential1
                t_potential1, t_potential2 = t_potential2, t_potential1

                # Load the next potential
                self.thr.to_device(self.potentials[p_index + 1], dest=potential2)
                t_potential2 = potential_dt * (p_index + 1)

            t = step * dt
            stepper(psi, psi, potential1, potential2, t_potential1, t_potential2, t)
            t = (step + 1) * dt

            if (step + 1) % (steps // self.samples) == 0:
                if half_step:
                    print(t, end=' ')
                    sys.stdout.flush()

                    res_dict, t_col = self._collect(psi, t, collectors)
                    results.append(res_dict)
                    t_collectors += t_col

                else:
                    print(t, end=' ')
                    sys.stdout.flush()

        print()

        t_total = time.time() - t_start

        print("Total time:", t_total, "s")
        if half_step:
            print("Collectors time:", t_collectors, "s")

        if half_step:
            return results
        else:
            res_dict, _ = self._collect(psi, t, collectors)
            return [res_dict]

    def __call__(self, psi, collectors):

        # double step (to estimate the convergence)
        psi_double = self.thr.copy_array(psi)
        results_double = self._integrate(psi_double, False, collectors)

        # actual integration
        results = self._integrate(psi, True, collectors)

        return IntegrationResults(psi_double.get(), psi.get(), results_double, results)


class IntegrationResults:

    def __init__(self, psi_double, psi, results_double, results):
        self._fill_errors(psi_double, psi, results_double, results)
        self._fill_results(results)

    def _fill_results(self, results):

        # Currently we have the results in form [{results for time t1}, {results for time t2}, ...]
        # We want to transpose them to look like
        # {result_group1: [result for time t1, result for time t2, ...], ...}

        self.values = {key:[] for key in results[0].keys()}
        for res in results:
            for key in res:
                self.values[key].append(res[key])

        # Stack the results into a numpy array, if the format allows
        for key in self.values:
            if isinstance(self.values[key][0], float):
                self.values[key] = numpy.array(self.values[key])
            elif isinstance(self.values[key][0], numpy.ndarray):
                self.values[key] = numpy.vstack(
                    [arr.reshape(1, *arr.shape) for arr in self.values[key]])

    def _fill_errors(self, psi_double, psi, results_double, results):
        # calculate the error (separately for each ensemble and component)
        psi_errors = self._batched_norm(psi_double - psi) / self._batched_norm(psi)

        self.errors = dict(psi_strong_mean=psi_errors.mean(), psi_strong_max=psi_errors.max())

        # calculate result errors
        final_results_double = results_double[-1]
        final_results = results[-1]
        for key in final_results:
            res_double = final_results_double[key]
            res = final_results[key]
            self.errors[key] = numpy.linalg.norm(res_double - res) / numpy.linalg.norm(res)

    def _batched_norm(self, x):
        norms = numpy.abs(x) ** 2

        # Sum over spatial dimensions
        norms = norms.reshape(x.shape[0], x.shape[1], product(x.shape[2:]))
        return numpy.sqrt(norms.sum(-1))

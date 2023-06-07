from .qubo import qubo
import numpy as np
from tqdm import tqdm
import time
import evoqapi


class DigitalAnnealing:

    def __init__(self, token, host='127.0.0.1'):
        # token of the user, fetchable from "Home" on the digital annealer website
        self.evoq = evoqapi.Backend(token, host)

    def sample_qubo(self, qubo, shots=1, anneal_time=1, initial_state=None, timeout=None,
                    **unknown_options):
        solutions, energies = self._schedule_shots(qubo, anneal_time, shots, initial_state, timeout)
        return solutions, energies

    def _schedule_shots(self, qubo, anneal_time, shots, initial_state, timeout):
        assert qubo.shape[0] <= 2048
        if timeout is not None:
            assert timeout > 2 * anneal_time * shots
        if initial_state is not None:
            assert initial_state.ndim == 1 and initial_state.shape[0] == qubo.shape[0]
        start = time.time()
        print('Waiting for all shots to be ready...')
        # schedule shots to digital annealer with annealing time 10e-5
        schedule = self.evoq.schedule(qubo, time=anneal_time, shots=shots,
                                      initial_state=initial_state)

        # wait until all shots are ready (poll with 10Hz)
        timeout_start = time.time()
        timeout_bool = False
        while not self.evoq.all_completed(schedule):
            if timeout is not None:
                if (time.time() - timeout_start) > timeout:
                    timeout_bool = True
                    break
            time.sleep(1.0 / 10.0)

        # collect all solutions
        # counts contains the corresponding counts
        end = time.time()
        if timeout_bool:
            print("Timeout: Not all shots are done!")
        else:
            print("All shots done in {} s!".format(end - start))
        start = time.time()
        print("Deleting shots and compute energies...")
        solutions = []
        for s in schedule:
            try:
                result = s.result
                if result is not None:
                    solutions += [result]
            except Exception:
                pass
        solutions = np.array(solutions)
        # delete shots
        for s in schedule:
            s.delete()

        energies = np.array([solution @ qubo @ solution for solution in solutions])
        end = time.time()
        print("Done in {} s!".format(end - start))
        return solutions, energies

    def sample_qubos(self, qubos, shots=1, anneal_time=1, timeout=None, batch_size=None,
                     **unknown_options):
        """
        Sample from a list of QUBOs.
        """
        solutions_list, energies_list = [], []
        schedule = []
        if batch_size is None:
            batch_size = len(qubos)
        if timeout is not None:
            assert timeout > 2 * anneal_time * shots
        print('Waiting for all QUBOs to be ready...')
        for batch_index in tqdm(range(0, len(qubos), batch_size), desc="Batch"):
            for qubo_index in tqdm(range(batch_index, min(batch_index + batch_size, len(qubos))),
                                   desc="Schedule QUBO"):
                qubo = qubos[qubo_index]
                assert qubo.shape[0] <= 2048
                for i in range(shots):
                    schedule.append(self.evoq.schedule_one(qubo, time=anneal_time))
            for qubo_index in tqdm(range(batch_index, min(batch_index + batch_size, len(qubos))),
                                   desc="QUBO number"):
                qubo = qubos[qubo_index]
                qubo_schedule = schedule[shots * qubo_index: shots * (qubo_index + 1)]
                timeout_start = time.time()
                timeout_bool = False
                while not self.evoq.all_completed(qubo_schedule):
                    if timeout is not None:
                        if (time.time() - timeout_start) > timeout:
                            timeout_bool = True
                            break
                    time.sleep(1.0 / 10.0)
                if timeout_bool:
                    print("Timeout: Not all shots are done!")
                solutions = []
                energies = []
                for s in qubo_schedule:
                    result = s.result
                    # print(result)
                    if result is not None:
                        result = np.array(result)
                        solutions.append(result)
                        energies.append(result @ qubo @ result)
                solutions = np.array(solutions)
                energies = np.array(energies)
                solutions_list.append(solutions)
                energies_list.append(energies)
        # delete shots
        for s in schedule:
            s.delete()
        return solutions_list, energies_list

import math
import random
import sys
import time
from collections import deque
from typing import Any, List, Tuple

from scipy.optimize import linear_sum_assignment


class Pos:
    def __init__(self, y: int, x: int):
        self.y = y
        self.x = x


class Judge:
    def set_temperature(self, temperature: List[List[int]]) -> None:
        for row in temperature:
            print(" ".join(map(str, row)))
        sys.stdout.flush()

    def measure(self, i: int, y: int, x: int) -> int:
        print(f"{i} {y} {x}", flush=True)
        v = int(input())
        if v == -1:
            print(f"something went wrong. i={i} y={y} x={x}", file=sys.stderr)
            sys.exit(1)
        return v

    def answer(self, estimate: List[int]) -> None:
        print("-1 -1 -1")
        for e in estimate:
            print(e)
        sys.stdout.flush()


class BaseTemperature:
    """base class for creating temperature distribution"""

    def __init__(self, L: int, N: int, S: int, landing_pos: List[Pos]):
        self.L = L
        self.N = N
        self.S = S
        self.landing_pos = landing_pos
        self.temperature = [[0] * self.L for _ in range(self.L)]


class BaseSolver:
    """base class for solving the problem"""

    def __init__(self, L: int, N: int, S: int, landing_pos: List[Pos]):
        self.L = L
        self.N = N
        self.S = S
        self.landing_pos = landing_pos
        self.judge = Judge()

    def solve(self) -> None:
        """solve the problem"""
        raise NotImplementedError

    def _hungarian_method(self, cost_matrix: List[List[float]], maximize: bool = False) -> List[int]:
        """optimize matching using hungarian method"""
        _, estimate = linear_sum_assignment(cost_matrix, maximize=maximize)
        return estimate


class OptimizeTemperature:
    def __init__(self, L: int, landing_pos: List[Pos]):
        self.L = L
        self.landing_pos = landing_pos

    def optimize(
        self,
        current_temperature: List[List[int]],
        T_start: float,
        T_end: float,
        alpha: float,
        prohibit_pos: List[Tuple[int, int]],
        rand: int = 5,
        time_limit: float = 3.33,
    ) -> List[List[int]]:
        """Optimize temperature distribution using simulated annealing"""
        start_time = time.time()
        best_temperature = [row[:] for row in current_temperature]
        best_cost = self._calc_placement_cost(current_temperature)
        permit_pos = [(i, j) for i in range(self.L) for j in range(self.L) if (i, j) not in prohibit_pos]
        if len(permit_pos) == 0:
            return best_temperature
        while time.time() - start_time <= time_limit:
            T = T_start
            while T > T_end:
                new_temperature = [row[:] for row in best_temperature]
                for _ in range(random.randint(1, 20)):
                    i, j = random.choice(permit_pos)
                    new_temperature[i][j] += random.randint(-rand, rand)
                    new_temperature[i][j] = max(0, min(1000, new_temperature[i][j]))

                neighbour_cost = self._calc_placement_cost(new_temperature)
                delta = neighbour_cost - best_cost
                if delta < 0 or random.uniform(0, 1) < math.exp(-delta / T):
                    best_temperature, best_cost = new_temperature, neighbour_cost

                T *= alpha

        return best_temperature

    def _calc_placement_cost(self, temperature: List[List[int]]) -> int:
        """calculate placement cost"""
        cost = 0
        for i in range(self.L):
            for j in range(self.L):
                cost += (temperature[i][j] - temperature[(i + 1) % self.L][j]) ** 2
                cost += (temperature[i][j] - temperature[i][(j + 1) % self.L]) ** 2
        return cost


class OnePointTemperature(BaseTemperature):
    def __init__(self, L: int, N: int, S: int, landing_pos: List[Pos]):
        super().__init__(L, N, S, landing_pos)
        self.optimizer = OptimizeTemperature(L, landing_pos)
        self.around = [(y, x) for y in range(-1, 2) for x in range(-1, 2) if not (y, x) == (0, 0)]
        self.around2 = [
            (y, x) for y in range(-2, 3) for x in range(-2, 3) if not (y, x) in self.around and not (y, x) == (0, 0)
        ]
        self.around3 = [
            (y, x)
            for y in range(-3, 4)
            for x in range(-3, 4)
            if not (y, x) in (self.around + self.around2) and not (y, x) == (0, 0)
        ]

    def run(self) -> List[List[int]]:
        interval = 2
        avg = (interval + self.N * interval) * self.N // 2 // self.N
        prohibit_pos: List[Tuple[int, int]] = [(pos.y, pos.x) for pos in self.landing_pos]
        center = Pos(self.L // 2, self.L // 2)
        center_distance = [{pos: abs(pos.y - center.y) + abs(pos.x - center.x)} for pos in self.landing_pos]
        center_distance = sorted(center_distance, key=lambda x: list(x.values())[0])
        for i in range(self.N):
            pos = list(center_distance[i].keys())[0]
            self.temperature[pos.y][pos.x] = interval * (i + 1)
            center_avg = 2 * (i + 1)
            avg_round1 = (3 * center_avg + avg) // 4
            avg_round2 = (center_avg + avg) // 2
            avg_round3 = (center_avg + 3 * avg) // 4
            for dy, dx in self.around:
                y, x = (pos.y + dy) % self.L, (pos.x + dx) % self.L
                if self.temperature[y][x] == 0:
                    self.temperature[y][x] = avg_round1
                elif self.temperature[y][x] != 0 and (y, x) not in prohibit_pos:
                    self.temperature[y][x] = (self.temperature[y][x] + avg_round1) // 2
            for dy, dx in self.around2:
                y, x = (pos.y + dy) % self.L, (pos.x + dx) % self.L
                if self.temperature[y][x] == 0:
                    self.temperature[y][x] = avg_round2
                elif self.temperature[y][x] != 0 and (y, x) not in prohibit_pos:
                    self.temperature[y][x] = (self.temperature[y][x] + avg_round2) // 2
            for dy, dx in self.around3:
                y, x = (pos.y + dy) % self.L, (pos.x + dx) % self.L
                if self.temperature[y][x] == 0:
                    self.temperature[y][x] = avg_round3
                elif self.temperature[y][x] != 0 and (y, x) not in prohibit_pos:
                    self.temperature[y][x] = (self.temperature[y][x] + avg_round3) // 2
        for i in range(self.L):
            for j in range(self.L):
                if self.temperature[i][j] == 0:
                    self.temperature[i][j] = avg

        rand = self.N // 20 - 2
        self.temperature = self.optimizer.optimize(
            current_temperature=self.temperature,
            T_start=100,
            T_end=0.1,
            alpha=0.1,
            prohibit_pos=prohibit_pos,
            rand=rand,
            time_limit=3.33,
        )
        return self.temperature


class OnePointSolver(BaseSolver):
    def __init__(self, L: int, N: int, S: int, landing_pos: List[Pos]):
        super().__init__(L, N, S, landing_pos)
        self.create_temperature = OnePointTemperature(L, N, S, landing_pos)
        print("# --- Selected OnePointSolver", file=sys.stderr)

    def solve(self) -> None:
        """solve the problem"""
        temperature = self.create_temperature.run()
        self.judge.set_temperature(temperature)
        estimate = self._predict(temperature)
        self.judge.answer(estimate)

    def _predict(self, temperature: List[List[int]]) -> List[int]:
        cost_matrix: List[List[float]] = [[0] * self.N for _ in range(self.N)]
        measure_count = 5
        for i_in in range(self.N - 1):
            measured_value = sum([self.judge.measure(i_in, 0, 0) for _ in range(measure_count)]) / measure_count
            for i_out, pos in enumerate(self.landing_pos):
                cost_matrix[i_in][i_out] = abs(temperature[pos.y][pos.x] - measured_value)
        estimate = self._hungarian_method(cost_matrix)
        return estimate


class SingularTemperature(BaseTemperature):
    def __init__(self, L: int, N: int, S: int, landing_pos: List[Pos]):
        super().__init__(L, N, S, landing_pos)

    def run(self) -> Tuple[List[List[int]], List[List[Any]]]:
        """return temperature distribution and distance from each i_out to singular point"""
        y, x = sum([pos.y for pos in self.landing_pos]) // self.N, sum([pos.x for pos in self.landing_pos]) // self.N
        self.temperature[y][x] = min(1000, 10 * self.S)
        distance: List[List[Any]] = [[] for _ in range(self.N)]
        for i_out, pos in enumerate(self.landing_pos):
            distance[i_out] = [i_out, [y - pos.y, x - pos.x], [abs(y - pos.y) + abs(x - pos.x)]]
        distance.sort(key=lambda x: x[2])
        return self.temperature, distance


class SingularSolver(BaseSolver):
    def __init__(self, L: int, N: int, S: int, landing_pos: List[Pos]):
        super().__init__(L, N, S, landing_pos)
        self.create_temperature = SingularTemperature(L, N, S, landing_pos)
        print("# --- Selected SingularSolver", file=sys.stderr)

    def solve(self) -> None:
        """solve the problem"""
        temperature, distance = self.create_temperature.run()
        self.judge.set_temperature(temperature)
        estimate = self._predict(temperature, distance)
        self.judge.answer(estimate)

    def _predict(self, temperature: List[List[int]], distance: List[List[Any]]) -> List[int]:
        """predict landing_pos"""
        threshold = min(999, 5 * self.S)
        estimate = [0] * self.N
        visited_index = []
        for i_in in range(self.N - 1):
            for i_out, [dy, dx], _ in distance:
                if i_out in visited_index:
                    continue
                if self.judge.measure(i_in, dy, dx) > threshold:
                    visited_index.append(i_out)
                    estimate[i_in] = i_out
                    break
        estimate[-1] = [i for i in range(self.N) if i not in visited_index][0]
        return estimate


class SingularTemperature2(BaseTemperature):
    def __init__(self, L: int, N: int, S: int, landing_pos: List[Pos]):
        super().__init__(L, N, S, landing_pos)

    def run(self) -> Tuple[List[List[int]], List[List[List[int]]]]:
        """return temperature distribution and distance from each i_out to singular point"""
        y, x = sum([pos.y for pos in self.landing_pos]) // self.N, sum([pos.x for pos in self.landing_pos]) // self.N
        self.temperature[y][x] = min(1000, 10 * self.S)
        self.temperature[(y + 1) % self.L][x] = min(1000, 10 * self.S)
        distance: List[List[List[int]]] = [[] for _ in range(self.N)]
        for i, pos in enumerate(self.landing_pos):
            distance[i] = [[y - pos.y, x - pos.x], [(y + 1) % self.L - pos.y, x - pos.x]]
        return self.temperature, distance


class SingularSolver2(BaseSolver):
    def __init__(self, L: int, N: int, S: int, landing_pos: List[Pos]):
        super().__init__(L, N, S, landing_pos)
        self.create_temperature = SingularTemperature2(L, N, S, landing_pos)

    def solve(self) -> None:
        """solve the problem"""
        temperature, distance = self.create_temperature.run()
        self.judge.set_temperature(temperature)
        estimate = self._predict(temperature, distance)
        self.judge.answer(estimate)

    def _predict(self, temperature: List[List[int]], distance: List[List[List[int]]]) -> List[int]:
        """predict landing_pos"""
        cost_matrix: List[List[float]] = [[0] * self.N for _ in range(self.N)]
        estimate = [0] * self.N
        if self.N <= 71:
            print("# --- Selected SinglarSolver2 ALL", file=sys.stderr)
            for i_in in range(self.N - 1):
                for i_out, [[dy, dx], [dy2, dx2]] in enumerate(distance):
                    cost_matrix[i_in][i_out] += self.judge.measure(i_in, dy, dx) + self.judge.measure(i_in, dy2, dx2)
            estimate = self._hungarian_method(cost_matrix, maximize=True)
        else:
            print("# --- Selected SingularSolver2 threhold", file=sys.stderr)
            threshold = 5.3 * self.S
            visited_index = []
            for i_in in range(self.N - 1):
                for i_out, [[dy, dx], [dy2, dx2]] in enumerate(distance):
                    if i_out in visited_index:
                        continue
                    if self.judge.measure(i_in, dy, dx) + self.judge.measure(i_in, dy2, dx2) > threshold:
                        estimate[i_in] = i_out
                        visited_index.append(i_out)
                        break
            estimate[-1] = [i for i in range(self.N) if i not in visited_index][0]
        return estimate


class AroundTemperature(BaseTemperature):
    def __init__(self, L: int, N: int, S: int, landing_pos: List[Pos]):
        super().__init__(L, N, S, landing_pos)
        self.optimizer = OptimizeTemperature(L, landing_pos)
        if self.S < 625:
            self.Around = 3
            self.direction = [(y, x) for y in range(-1, 2) for x in range(-1, 2)]
            if self.S >= 289 and self.N == 100:
                self.Around = 5
                self.direction = [(y, x) for y in range(-2, 3) for x in range(-2, 3)]
        else:
            self.Around = 5
            self.direction = [(y, x) for y in range(-2, 3) for x in range(-2, 3)]

    def run(self) -> Tuple[List[List[int]], List[List[float]]]:
        avg, num = 0, 0
        prohibit_pos: List[Tuple[int, int]] = [(pos.y, pos.x) for pos in self.landing_pos]
        center_temps = [min(1000, max(0, int(random.gauss(500, self.S * 2)))) for _ in range(self.N)]
        center_temps.sort()
        landing_deque = deque(self.landing_pos)
        for i in range(self.N):
            center_temp = center_temps[i]
            if i % 2 == 0:
                pos = landing_deque.popleft()
            else:
                pos = landing_deque.pop()
            self.temperature[pos.y][pos.x] = center_temp
            avg += self.temperature[pos.y][pos.x]
            num += 1
            for dy, dx in self.direction:
                y, x = (pos.y + dy) % self.L, (pos.x + dx) % self.L
                prohibit_pos.append((y, x))
                if self.temperature[y][x] == 0:
                    if self.S * self.N <= 26460:
                        self.temperature[y][x] = min(1000, max(0, int(random.gauss(center_temp, self.S * 0.9))))
                    else:
                        self.temperature[y][x] = min(1000, max(0, int(random.gauss(center_temp, self.S))))
                    avg += self.temperature[y][x]
                    num += 1
        avg //= num
        for i in range(self.L):
            for j in range(self.L):
                if self.temperature[i][j] == 0:
                    self.temperature[i][j] = avg
        prohibit_pos = list(set(prohibit_pos))
        self.temperature = self.optimizer.optimize(
            current_temperature=self.temperature,
            T_start=100,
            T_end=0.1,
            alpha=0.1,
            prohibit_pos=prohibit_pos,
            rand=3,
            time_limit=3.33,
        )

        vector: List[List[float]] = [[] for _ in range(self.N)]
        for i, pos in enumerate(self.landing_pos):
            for dy, dx in self.direction:
                vector[i].append(self.temperature[(pos.y + dy) % self.L][(pos.x + dx) % self.L])
        return self.temperature, vector


class AroundSolver(BaseSolver):
    def __init__(self, L: int, N: int, S: int, landing_pos: List[Pos]):
        super().__init__(L, N, S, landing_pos)
        self.create_temperature = AroundTemperature(L, N, S, landing_pos)
        if self.S < 625:
            self.Around = 3
            self.direction = [(y, x) for y in range(-1, 2) for x in range(-1, 2)]
            if self.S >= 289 and self.N == 100:
                self.Around = 5
                self.direction = [(y, x) for y in range(-2, 3) for x in range(-2, 3)]
        else:
            self.Around = 5
            self.direction = [(y, x) for y in range(-2, 3) for x in range(-2, 3)]
        print(f"# --- Selected {self.Around} AroundSolver", file=sys.stderr)

    def solve(self) -> None:
        """solve the problem"""
        temperature, vector = self.create_temperature.run()
        self.judge.set_temperature(temperature)
        estimate = self._predict(temperature, vector)
        self.judge.answer(estimate)

    def _predict(self, temperature: List[List[int]], vector: List[List[float]]) -> List[int]:
        cost_matrix: List[List[float]] = [[0] * self.N for _ in range(self.N)]
        measure_count = 10000 // self.N // self.Around**2
        print(f"# measure_count={measure_count}", file=sys.stderr)
        for i_in in range(self.N):
            print(f"# measure i={i_in} y=0 x=0")
            measure_vector = []
            for dy, dx in self.direction:
                measure_vector.append(
                    sum([self.judge.measure(i_in, dy, dx) for _ in range(measure_count)]) / measure_count
                )
            for i_out, pos in enumerate(self.landing_pos):
                cost: float = 0
                for i in range(len(measure_vector)):
                    cost += ((vector[i_out][i] - measure_vector[i])) ** 2
                cost_matrix[i_in][i_out] = cost
        estimate = self._hungarian_method(cost_matrix)
        return estimate


def main():
    L, N, S = map(int, input().split())
    print(f"# L={L} N={N} S={S}", file=sys.stderr)
    landing_pos = []
    for _ in range(N):
        y, x = map(int, input().split())
        landing_pos.append(Pos(y, x))
    if S == 1:
        solver = OnePointSolver(L, N, S, landing_pos)
    elif 1 < S < 169:
        solver = SingularSolver(L, N, S, landing_pos)
    elif 169 <= S < 289:
        solver = SingularSolver2(L, N, S, landing_pos)
    elif 289 <= S < 400 and N <= 71:
        solver = SingularSolver2(L, N, S, landing_pos)
    else:
        solver = AroundSolver(L, N, S, landing_pos)
    solver.solve()


if __name__ == "__main__":
    main()

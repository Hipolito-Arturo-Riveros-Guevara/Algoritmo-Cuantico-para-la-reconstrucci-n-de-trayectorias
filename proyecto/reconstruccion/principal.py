from proyecto.componentes.componentes import Evento, Segmento
from proyecto.reconstruccion.hamiltoniano import Hamiltoniano
from itertools import product, count
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import cg
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit_optimization import QuadraticProgram
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ajustar_potencia2(A, b):
    m = A.shape[0]
    d = int(2**np.ceil(np.log2(m)) - m)
    if d > 0:
        A_tilde = np.block([[A, np.zeros((m, d), dtype=np.float64)], [np.zeros((d, m), dtype=np.float64), np.eye(d, dtype=np.float64)]])
        b_tilde = np.block([b, b[:d]])
        return A_tilde, b_tilde
    else:
        return A, b

class HamiltonianoSimple(Hamiltoniano):
    def __init__(self, epsilon, gamma, delta):
        self.epsilon = epsilon
        self.gamma = gamma
        self.delta = delta
        self.A = None
        self.M = None
        self.D = None
        self.b = None
        self.segmentos = None
        self.segmentos_agrupados = None
        self.n_segmentos = None

    def construir_segmentos(self, evento: Evento):
        segmentos_agrupados = []
        segmentos = []
        n_segmentos = 0
        id_segmento = count()
        for idx in range(len(evento.modulos) - 1):
            desde_impactos = evento.modulos[idx].impactos
            hacia_impactos = evento.modulos[idx + 1].impactos

            grupo_segmentos = []
            for desde_impacto, hacia_impacto in product(desde_impactos, hacia_impactos):
                seg = Segmento(next(id_segmento), desde_impacto, hacia_impacto)
                grupo_segmentos.append(seg)
                segmentos.append(seg)
                n_segmentos += 1

            segmentos_agrupados.append(grupo_segmentos)

        self.segmentos_agrupados = segmentos_agrupados
        self.segmentos = segmentos
        self.n_segmentos = n_segmentos

    def construir_hamiltoniano(self, evento: Evento):

        if self.segmentos_agrupados is None:
            self.construir_segmentos(evento)

        A = eye(self.n_segmentos, format='lil') * (-(self.delta + self.gamma))
        b = np.ones(self.n_segmentos) * self.delta

        for idx_grupo in range(len(self.segmentos_agrupados) - 1):
            for seg_i, seg_j in product(self.segmentos_agrupados[idx_grupo], self.segmentos_agrupados[idx_grupo + 1]):
                if seg_i.impacto_hasta == seg_j.impacto_desde:
                    coseno = seg_i * seg_j

                    if abs(coseno - 1) < self.epsilon:
                        A[seg_i.id_segmento, seg_j.id_segmento] = A[seg_j.id_segmento, seg_i.id_segmento] = 1

        A = A.tocsc()

        self.A, self.b = -A, b
        return -A, b

    def hamiltoniano_evento(self, evento: Evento):

        if self.segmentos_agrupados is None:
            self.construir_segmentos(evento)

        A = eye(self.n_segmentos, format='lil') * (-(self.delta + self.gamma))
        b = np.ones(self.n_segmentos) * self.delta

        for idx_grupo in range(len(self.segmentos_agrupados) - 1):
            for seg_i, seg_j in product(self.segmentos_agrupados[idx_grupo], self.segmentos_agrupados[idx_grupo + 1]):
                if seg_i.impacto_hasta == seg_j.impacto_desde:
                    coseno = seg_i * seg_j

                    if abs(coseno - 1) < self.epsilon:
                        A[seg_i.id_segmento, seg_j.id_segmento] = A[seg_j.id_segmento, seg_i.id_segmento] = 1

        A = A.tocsc()

        self.A, self.b = -A, b
        return cg(self.A, self.b, atol=0)

    def isign_hamiltoniano(self, evento: Evento):
        if self.segmentos_agrupados is None:
            self.construir_segmentos(evento)

        M = csr_matrix((self.n_segmentos, self.n_segmentos))

        for idx_grupo in range(len(self.segmentos_agrupados) - 1):
            for seg_i, seg_j in product(self.segmentos_agrupados[idx_grupo], self.segmentos_agrupados[idx_grupo + 1]):
                if seg_i.impacto_hasta == seg_j.impacto_desde:
                    coseno = seg_i * seg_j

                    if abs(coseno - 1) < self.epsilon:
                        M[seg_i.id_segmento, seg_j.id_segmento] = 1
                        M[seg_j.id_segmento, seg_i.id_segmento] = 1
        M = M.tocsc()
        self.M = M
        return M

    def hamiltoniano_dp(self, evento: Evento):
        if self.segmentos_agrupados is None:
            self.construir_segmentos(evento)

        qubo = QuadraticProgram()
        costo = {}
        D = np.zeros((self.n_segmentos, self.n_segmentos))

        for i in range(self.n_segmentos):
            qubo.binary_var(name='x' + str(i))

        for idx_grupo in range(len(self.segmentos_agrupados) - 1):
            for seg_i, seg_j in product(self.segmentos_agrupados[idx_grupo], self.segmentos_agrupados[idx_grupo + 1]):
                if seg_i.impacto_hasta == seg_j.impacto_desde:
                    coseno = seg_i * seg_j
                    vi = seg_i.a_vector()
                    vj = seg_j.a_vector()

                    ri = (vi[0]**2 + vi[1]**2 + vi[2]**2)**0.5
                    rj = (vj[0]**2 + vj[1]**2 + vj[2]**2)**0.5

                    ang = -0.5 * np.power(coseno, 3) / (ri + rj)

                    if ang != 0:
                        xi = 'x' + str(seg_i.id_segmento)
                        xj = 'x' + str(seg_j.id_segmento)
                        costo[(xi, xj)] = ang
                        D[seg_i.id_segmento, seg_j.id_segmento] = D[seg_j.id_segmento, seg_i.id_segmento] = ang

        for idx_grupo in range(len(self.segmentos_agrupados)):
            for seg_i, seg_j in product(self.segmentos_agrupados[idx_grupo], self.segmentos_agrupados[idx_grupo]):
                if (seg_i.impacto_desde.id_impacto == seg_j.impacto_desde.id_impacto and
                        seg_i.impacto_hasta.id_impacto != seg_j.impacto_hasta.id_impacto):
                    xi = 'x' + str(seg_i.id_segmento)
                    xj = 'x' + str(seg_j.id_segmento)
                    costo[(xi, xj)] = 0.5
                    D[seg_i.id_segmento, seg_j.id_segmento] = D[seg_j.id_segmento, seg_i.id_segmento] = 0.5

        self.D = D
        qubo.minimize(quadratic=costo)
        return qubo, costo

    def qu_solv(self):
        if self.A is None:
            raise Exception("No inicializado")

        solucion, _ = cg(self.A, self.b, atol=0)
        return solucion

    def visualizar_evento(self, evento: Evento):
        X = np.array([obj.x for obj in evento.impactos])
        Y = np.array([obj.y for obj in evento.impactos])
        Z = np.array([obj.z for obj in evento.impactos])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, c='b', marker='o')

        x_range = np.linspace(X.min(), X.max(), 10)
        y_range = np.linspace(Y.min(), Y.max(), 10)
        Xp, Yp = np.meshgrid(x_range, y_range)

        planos_z = list(range(1, len(evento.modulos) + 1))

        for plano_z in planos_z:
            Zp = np.full(Xp.shape, plano_z)
            ax.plot_surface(Xp, Yp, Zp, color='cyan', alpha=0.3, rstride=100, cstride=100, edgecolor='none')

        ax.grid(False)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def evaluar(self, solucion):
        if self.A is None:
            raise Exception("No inicializado")

        if isinstance(solucion, list):
            sol = np.array([solucion, None])
        elif isinstance(solucion, np.ndarray):
            if solucion.ndim == 1:
                sol = solucion[..., None]
            else:
                sol = solucion

        return -0.5 * sol.T @ self.A @ sol + self.b.dot(sol)

import numpy as np
from proyecto.componentes.componentes import Evento, Segmento
import proyecto.componentes as comp
import dataclasses
from itertools import count
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
@dataclasses.dataclass(frozen=True)
class Geometria_detector:
    id_modulos   : list[int]
    lx           : list[float]
    ly           : list[float]
    z            : list[float]
    
    def __getitem__(self, indice):
        return (self.id_modulos[indice], self.lx[indice], self.ly[indice], self.z[indice])
    
    def __len__(self):
        return len(self.id_modulos)
    
 
@dataclasses.dataclass()   
class Generador:
    geometria_detector   : Geometria_detector
    phi_min              : float = 0.0
    phi_max              : float = 2*np.pi
    theta_min            : float = 0.0
    theta_max            : float = np.pi/10
    vertice_primario     : tuple = (0.0, 0.0, 0.0)
    generador_aleatorio  : np.random.Generator = np.random.default_rng()
    
    def generar_evento(self, n_particulas):
        contador_id_impacto = count()
        info_mc = []
        
        impactos_por_modulo = [[] for _ in range(len(self.geometria_detector.id_modulos))]
        impactos_por_rastro  = []
        
        pvx, pvy, pvz = self.vertice_primario
        
        for id_rastro in range(n_particulas):
            phi         = self.generador_aleatorio.uniform(self.phi_min, self.phi_max)
            cos_theta   = self.generador_aleatorio.uniform(np.cos(self.theta_max), np.cos(self.theta_min))
            theta = np.arccos(cos_theta)
            sin_theta = np.sin(theta)
            
            info_mc.append((id_rastro, comp.InfoMC(
                self.vertice_primario,
                phi,
                theta)))
            
            vx = sin_theta * np.cos(phi)
            vy = sin_theta * np.sin(phi)
            vz = cos_theta
            
            impactos_rastro = []
            for idx, (id_modulo, zm, lx, ly) in enumerate(zip(self.geometria_detector.id_modulos, self.geometria_detector.z, self.geometria_detector.lx, self.geometria_detector.ly)):
                t = (zm - pvz) / vz
                x_impacto = pvx + vx * t
                y_impacto = pvy + vy * t
                
                if np.abs(x_impacto) < lx / 2 and np.abs(y_impacto) < ly / 2:
                    impacto = comp.Impacto(next(contador_id_impacto), x_impacto, y_impacto, zm, id_modulo, id_rastro)
                    impactos_por_modulo[idx].append(impacto)
                    impactos_rastro.append(impacto)
            impactos_por_rastro.append(impactos_rastro)
        
        modulos = [comp.Modulo(id_modulo, z, lx, ly, impactos_por_modulo[idx]) for idx, (id_modulo, z, lx, ly) in enumerate(zip(self.geometria_detector.id_modulos, self.geometria_detector.z, self.geometria_detector.lx, self.geometria_detector.ly))]
        rastros = []
        
        for idx, (id_rastro, info_mc) in enumerate(info_mc):
            rastros.append(comp.Rastro(id_rastro, info_mc, impactos_por_rastro[idx]))
        impactos_globales = [impacto for sublista in impactos_por_modulo for impacto in sublista]
        print(impactos_globales)
        return comp.Evento(modulos, rastros, impactos_globales)

    def visualizar(self, evento):
        N_MODULOS = len(self.geometria_detector.id_modulos)
        X = np.array([obj.x for obj in evento.impactos])
        Y = np.array([obj.y for obj in evento.impactos])
        Z = np.array([obj.z for obj in evento.impactos])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, c='b', marker='o')
        
        x_range = np.linspace(X.min(), X.max(), 10)
        y_range = np.linspace(Y.min(), Y.max(), 10)
        Xp, Yp = np.meshgrid(x_range, y_range)
        
        planos_z = list(range(1, N_MODULOS + 1))
        
        for plano_z in planos_z:
            Zp = np.full(Xp.shape, plano_z)
            ax.plot_surface(Xp, Yp, Zp, color='cyan', alpha=0.3, rstride=100, cstride=100, edgecolor='none')
        
        ax.grid(False)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.show()

    def visualizar_solucion(self, evento, solucion):
        X = np.array([obj.x for obj in evento.impactos])
        Y = np.array([obj.y for obj in evento.impactos])
        Z = np.array([obj.z for obj in evento.impactos])
        N_MODULOS = len(self.geometria_detector.id_modulos)
        capas = []
        for i in range(N_MODULOS):
            capa = [obj for obj in evento.impactos if obj.id_modulo == i]
            capas.append(capa)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X, Y, Z, c='b', marker='o')

        l = 0
        for i in range(N_MODULOS - 1):
            for j in capas[i]:
                for k in capas[i + 1]:
                    if solucion[l] == 1: 
                        ax.plot([j.x, k.x], [j.y, k.y], [j.z, k.z], 'r-', color='red', linewidth=0.7)
                    l += 1

        x_range = np.linspace(X.min(), X.max(), 10)
        y_range = np.linspace(Y.min(), Y.max(), 10)
        Xp, Yp = np.meshgrid(x_range, y_range)

        planos_z = list(range(1, N_MODULOS + 1))

        for plano_z in planos_z:
            Zp = np.full(Xp.shape, plano_z)
            ax.plot_surface(Xp, Yp, Zp, color='cyan', alpha=0.3, rstride=100, cstride=100, edgecolor='none')

        ax.grid(False)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

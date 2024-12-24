from abc import ABC, abstractmethod
from proyecto.componentes.componentes import Evento

class Hamiltoniano(ABC):
    
    @abstractmethod
    def construir_hamiltoniano(self, evento: Evento):
        pass
    
    @abstractmethod
    def evaluar(self, solucion):
        pass
